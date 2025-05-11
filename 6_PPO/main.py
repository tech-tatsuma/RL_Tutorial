import argparse
import os
import shutil
import gym
import imageio
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# Transition 定義
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'log_prob', 'next_state'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'running_reward'])

# ============================
# Actor ネットワーク
# ============================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 8)
        self.mu_head = nn.Linear(8, action_dim)
        self.sigma_head = nn.Linear(8, action_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        mu = self.mu_head(x)
        sigma = F.softplus(self.sigma_head(x)) + 1e-5
        return mu, sigma

# ============================
# Critic ネットワーク
# ============================
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 8)
        self.value_head = nn.Linear(8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        value = self.value_head(x)
        return value

# ============================
# PPO アルゴリズム
# ============================
class PPO:
    def __init__(self, state_dim, action_dim, args):
        self.actor_net = Actor(state_dim, action_dim).to(args.device)
        self.critic_net = Critic(state_dim).to(args.device)
        self.buffer = []
        self.buffer_capacity = args.buffer_capacity
        self.batch_size = args.batch_size
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.gamma = args.gamma
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic
        self.max_grad_norm = args.max_grad_norm
        self.device = args.device

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr_critic)
        self.counter = 0

    def select_action(self, state):
        state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            mu, sigma = self.actor_net(state_t)
        dist = Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        action_clamped = action.clamp(-2.0, 2.0)  # 環境に合わせて調整
        return action_clamped.cpu().numpy()[0], log_prob.cpu().item()

    def store_transition(self, trans):
        self.buffer.append(trans)
        self.counter += 1
        if self.counter >= self.buffer_capacity:
            return True
        return False

    def update(self):
        # バッファからテンソルを構築
        states = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(self.device)
        actions = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1,1).to(self.device)
        rewards = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float).to(self.device)
        old_log_probs = torch.tensor([t.log_prob for t in self.buffer], dtype=torch.float).view(-1,1).to(self.device)

        # リターンとアドバンテージ計算
        returns = rewards.clone()
        for i in reversed(range(len(rewards)-1)):
            returns[i] = rewards[i] + self.gamma * returns[i+1]
        advantage = (returns - self.critic_net(states)).detach()

        # PPO 更新
        for _ in range(self.ppo_epoch):
            sampler = BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, drop_last=False)
            for idx in sampler:
                mu, sigma = self.actor_net(states[idx])
                dist = Normal(mu, sigma)
                log_probs = dist.log_prob(actions[idx]).sum(dim=1, keepdim=True)
                ratio = torch.exp(log_probs - old_log_probs[idx])
                surr1 = ratio * advantage[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(states[idx]), returns[idx])
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        # バッファクリア
        self.buffer.clear()
        self.counter = 0

# ============================
# 動画記録ユーティリティ
# ============================
def record_episode(agent, env, max_steps, device):
    frames = []
    state, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        action, _ = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
        steps += 1
    return frames, steps

# ============================
# メイン
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v1')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--buffer_capacity', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ppo_epoch', type=int, default=10)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--lr_critic', type=float, default=4e-3)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--max_eps', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--reward_win', type=float, default=-200.0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--save_prefix', type=str, default='ppo')
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make(args.env, render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim, args)
    os.makedirs('temp_videos', exist_ok=True)
    demo_paths = []
    rewards = []
    best_avg = -np.inf
    no_improve = 0

    for ep in range(1, args.max_eps+1):
        state, _ = env.reset()
        score = 0
        for t in range(1, 1001):
            action, logp = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.append(Transition(state, action, reward, logp, next_state))
            state = next_state
            score += reward
            if len(agent.buffer) >= args.buffer_capacity:
                agent.update()
            if done:
                break

        rewards.append(score)
        # 移動平均更新
        if ep == 1:
            running = score
        else:
            running = 0.9 * running + 0.1 * score

        # ログ出力
        if ep % args.log_interval == 0:
            print(f"Episode {ep}\tScore: {score:.2f}\tRunning: {running:.2f}")
            frames, _ = record_episode(agent, env, max_steps=200, device=args.device)
            path = f"temp_videos/ep{ep}.mp4"
            imageio.mimsave(path, frames, fps=args.fps)
            demo_paths.append(path)

        # Early stopping
        if ep >= 100:
            avg100 = np.mean(rewards[-100:])
            if avg100 > best_avg:
                best_avg = avg100
                no_improve = 0
                torch.save(agent.actor_net.state_dict(), f"{args.save_prefix}_actor.pth")
                torch.save(agent.critic_net.state_dict(), f"{args.save_prefix}_critic.pth")
            else:
                no_improve += 1
            if best_avg >= args.reward_win or no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    # 学習曲線保存
    plt.figure()
    plt.plot(rewards, label='score')
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), label='avg100')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('learning_curve.png')

    # 動画結合
    writer = imageio.get_writer('training_progress.mp4', fps=args.fps)
    for p in demo_paths:
        for frame in imageio.get_reader(p): writer.append_data(frame)
    writer.close()
    shutil.rmtree('temp_videos')

    # 最終デモ
    frames, steps = record_episode(agent, env, max_steps=200, device=args.device)
    imageio.mimsave('final_demo.mp4', frames, fps=args.fps)
    print(f"Saved final_demo.mp4 ({steps} steps)")

if __name__ == '__main__':
    main()
