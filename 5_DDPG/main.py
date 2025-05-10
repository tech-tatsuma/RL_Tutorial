import argparse
import os
import shutil
import gym
import imageio
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ============================
# Ornstein–Uhlenbeck ノイズ
# ============================
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.dim)
        self.state += dx
        return self.state

# ============================
# 経験リプレイバッファ
# ============================
class ReplayBuffer:
    def __init__(self, capacity):
        self.storage = []
        self.capacity = capacity
        self.ptr = 0

    def push(self, data):
        if len(self.storage) < self.capacity:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.capacity

    def __len__(self):
        return len(self.storage)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []
        for i in ind:
            s, s2, a, r, d = self.storage[i]
            states.append(s)
            next_states.append(s2)
            actions.append(a)
            rewards.append(r)
            dones.append(d)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(next_states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(1 - np.array(dones)).unsqueeze(1)
        )

# ============================
# アクターネットワーク（連続アクション出力）
# ============================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

# ============================
# クリティックネットワーク（Q値を予測）
# ============================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        x1 = F.relu(self.l1(xu))
        x2 = F.relu(self.l2(x1))
        return self.l3(x2)

# ============================
# DDPG エージェント
# ============================
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, args):
        device = args.device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=args.lr * 5)

        self.replay = ReplayBuffer(args.capacity)
        self.gamma = args.gamma
        self.tau   = args.tau
        self.batch_size = args.batch_size
        self.device = device
        self.noise  = OUNoise(action_dim)

    def select_action(self, state, noise=True):
        """ノイズ付きでアクション選択"""
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state_t).cpu().data.numpy().flatten()
        if noise:
            action += self.noise.sample()
        return action.clip(env.action_space.low, env.action_space.high)

    def train(self, iters):
        for _ in range(iters):
            if len(self.replay) < self.batch_size:
                return
            s, s2, a, r, d = self.replay.sample(self.batch_size)
            s, s2, a, r, d = [t.to(self.device) for t in (s, s2, a, r, d)]

            # Critic update
            with torch.no_grad():
                target_Q = self.critic_target(s2, self.actor_target(s2))
                target_Q = r + d * self.gamma * target_Q
            current_Q = self.critic(s, a)
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            actor_loss = -self.critic(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # ターゲットネットのソフト更新
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

# ============================
# エピソードのプレイ内容を記録
# ============================
def record_episode(agent, env, max_steps=1000):
    frames = []
    state, _ = env.reset()
    agent.noise.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = agent.select_action(state, noise=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        state = next_state
        done = terminated or truncated
        steps += 1
    return frames, steps

# ============================
# メインルーチン
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',               type=str,   default='Pendulum-v1')
    parser.add_argument('--mode',              type=str,   default='train')
    parser.add_argument('--max_eps',           type=int,   default=500)
    parser.add_argument('--capacity',          type=int,   default=100000)
    parser.add_argument('--batch_size',        type=int,   default=64)
    parser.add_argument('--gamma',             type=float, default=0.99)
    parser.add_argument('--tau',               type=float, default=0.005)
    parser.add_argument('--lr',                type=float, default=1e-4)
    parser.add_argument('--exploration_noise', type=float, default=0.2)
    parser.add_argument('--sample_freq',       type=int,   default=1000)
    parser.add_argument('--update_iters',      type=int,   default=50)
    parser.add_argument('--log_interval',      type=int,   default=10)
    parser.add_argument('--patience',          type=int,   default=20)
    parser.add_argument('--reward_win',        type=float, default=-200.0)
    parser.add_argument('--fps',               type=int,   default=30)
    parser.add_argument('--save_prefix',       type=str,   default='ddpg')
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make(args.env, render_mode='rgb_array')
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action, args)

    os.makedirs("temp_videos", exist_ok=True)
    demo_video_paths = []
    rewards = []
    best_avg = -np.inf
    no_improve = 0
    total_steps = 0

    for ep in range(1, args.max_eps+1):
        state, _ = env.reset()
        agent.noise.reset()
        ep_reward = 0

        while True:
            action = agent.select_action(state, noise=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay.push((state, next_state, action, reward, done))
            state = next_state
            ep_reward += reward
            total_steps += 1

            # サンプル頻度に達したらまとめて update
            if total_steps % args.sample_freq == 0:
                agent.train(args.update_iters)

            if done:
                break

        rewards.append(ep_reward)

        # ログ & early stopping 判定
        if ep >= 50:
            avg50 = np.mean(rewards[-50:])
            if ep % args.log_interval == 0:
                print(f"Episode {ep}\tReward {ep_reward:.2f}\tAvg50 {avg50:.2f}")
                # 動画記録
                frames, _ = record_episode(agent, env)
                path = f"temp_videos/ep{ep}.mp4"
                imageio.mimsave(path, frames, fps=args.fps)
                demo_video_paths.append(path)

            if avg50 > best_avg:
                best_avg = avg50
                no_improve = 0
                torch.save(agent.actor.state_dict(), f"{args.save_prefix}_actor.pth")
                torch.save(agent.critic.state_dict(), f"{args.save_prefix}_critic.pth")
            else:
                no_improve += 1

            if best_avg >= args.reward_win or no_improve >= args.patience:
                print("Early stopping.")
                break

    # 学習曲線保存
    plt.figure()
    plt.plot(rewards, label='ep reward')
    plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), label='avg50')
    plt.legend(); plt.savefig('learning_curve.png')

    # 動画結合
    writer = imageio.get_writer("training_progress.mp4", fps=args.fps)
    for vp in demo_video_paths:
        reader = imageio.get_reader(vp)
        for frame in reader: writer.append_data(frame)
        reader.close()
    writer.close()
    shutil.rmtree("temp_videos")

    # 最終デモ
    final_vid, final_steps = record_episode(agent, env)
    imageio.mimsave("final_demo.mp4", final_vid, fps=args.fps)
    print(f"Saved final_demo.mp4 ({final_steps} steps)")