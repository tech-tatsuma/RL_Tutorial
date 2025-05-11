import argparse
import os
import random
from collections import namedtuple, deque
from itertools import count

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imageio
import matplotlib.pyplot as plt
import shutil

# 動画記録用ユーティリティ
def record_episode(agent, env, max_steps=200, fps=30):
    frames = []
    state, _ = env.reset()
    for _ in range(max_steps):
        action = agent.select_action(state)
        state, reward, done, truncated, _ = env.step(action)
        frames.append(env.render())
        if done or truncated:
            break
    return frames

# デバイス設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = eps

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

# 引数パーサ
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
parser.add_argument("--env_name", default="Pendulum-v1")
parser.add_argument('--tau', default=0.005, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--capacity', default=50000, type=int)
parser.add_argument('--num_iteration', default=10000, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--render', default=False, type=bool)
parser.add_argument('--patience_interval', default=50, type=int)
parser.add_argument('--policy_noise', default=0.2, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
# PER用パラメータ
parser.add_argument('--per_alpha', default=0.6, type=float, help='優先度付き経験再生のalpha')
parser.add_argument('--per_beta_start', default=0.4, type=float, help='重要度サンプリングweightのbeta開始値')
parser.add_argument('--per_beta_frames', default=100000, type=int, help='betaが1.0に到達するまでのフレーム数')
args = parser.parse_args()

# 環境・シード設定
env = gym.make(args.env_name, render_mode='rgb_array')
torch.manual_seed(args.seed)
numpy_rng = np.random.RandomState(args.seed)
random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# 状態正規化インスタンスのみ
state_rms = RunningMeanStd((state_dim,))

# Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, frame_idx):
        prios = self.priorities if len(self.buffer) == self.capacity else self.priorities[:self.pos]
        P = prios ** self.alpha
        P /= P.sum()

        indices = numpy_rng.choice(len(self.buffer), batch_size, p=P)
        transitions = [self.buffer[idx] for idx in indices]
        beta = min(1.0, args.per_beta_start + frame_idx * (1.0 - args.per_beta_start) / args.per_beta_frames)
        weights = (len(self.buffer) * P[indices]) ** (-beta)
        weights /= weights.max()
        states, next_states, actions, rewards, dones = zip(*transitions)
        # 状態だけ正規化
        state_arr = state_rms.normalize(np.array(states))
        next_state_arr = state_rms.normalize(np.array(next_states))
        batch = (
            torch.FloatTensor(state_arr).to(device),
            torch.FloatTensor(next_state_arr).to(device),
            torch.FloatTensor(actions).to(device),
            torch.FloatTensor(rewards).unsqueeze(1).to(device),
            torch.FloatTensor(dones).unsqueeze(1).to(device),
        )
        return batch, indices, torch.FloatTensor(weights).unsqueeze(1).to(device)

    def update_priorities(self, indices, errors, eps=1e-6):
        for idx, err in zip(indices, errors.detach().cpu().numpy()):
            self.priorities[idx] = abs(err) + eps

# Actor-Critic 定義
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action
    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        return torch.tanh(self.fc3(a)) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.fc1(sa))
        q = F.relu(self.fc2(q))
        return self.fc3(q)

class TD3_PER:
    def __init__(self):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=args.learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=args.learning_rate)
        self.memory = PrioritizedReplayBuffer(args.capacity, args.per_alpha)

    def select_action(self, state):
        norm_s = state_rms.normalize(np.array(state).reshape(1, -1))
        s = torch.FloatTensor(norm_s).to(device)
        return self.actor(s).cpu().data.numpy().flatten()

    def update(self, it, frame_idx):
        (state, next_state, action, reward, done), indices, weights = self.memory.sample(args.batch_size, frame_idx)
        noise = (torch.randn_like(action) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-max_action, max_action)
        target_Q1 = self.critic_1_target(next_state, next_action)
        target_Q2 = self.critic_2_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1 - done) * args.gamma * target_Q.detach()
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        td_error1 = current_Q1 - target_Q
        td_error2 = current_Q2 - target_Q
        loss_Q1 = (weights * td_error1.pow(2)).mean()
        loss_Q2 = (weights * td_error2.pow(2)).mean()
        self.critic_1_optimizer.zero_grad(); loss_Q1.backward(); self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad(); loss_Q2.backward(); self.critic_2_optimizer.step()
        errors = (td_error1.abs() + td_error2.abs()) / 2
        self.memory.update_priorities(indices, errors)
        if it % args.policy_delay == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(tp.data * (1-args.tau) + p.data * args.tau)
            for p, tp in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                tp.data.copy_(tp.data * (1-args.tau) + p.data * args.tau)
            for p, tp in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                tp.data.copy_(tp.data * (1-args.tau) + p.data * args.tau)

    def save(self, prefix=""):
        os.makedirs(prefix + "models", exist_ok=True)
        torch.save(self.actor.state_dict(), prefix + "models/actor.pth")
        print("Model saved to", prefix + "models/")

    def load(self, prefix=""):
        self.actor.load_state_dict(torch.load(prefix + "models/actor.pth"))
        print("Model loaded from", prefix + "models/")

# メイン
if __name__ == '__main__':
    agent = TD3_PER()
    rewards = []
    demo_video_paths = []
    best_avg = -np.inf
    no_improve = 0
    os.makedirs('temp_videos', exist_ok=True)

    for ep in range(args.num_iteration):
        state, _ = env.reset()
        ep_r = 0
        for t in range(200):
            action = agent.select_action(state)
            action = (action + np.random.normal(0, args.exploration_noise, size=action_dim))
            action = action.clip(env.action_space.low, env.action_space.high)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.memory.push((state, next_state, action, reward, float(done)))
            state = next_state
            ep_r += reward
            if done or truncated:
                break
        rewards.append(ep_r)
        if len(agent.memory.buffer) >= args.batch_size:
            agent.update(ep, ep)
        if ep % args.log_interval == 0 and ep > 0:
            avg = np.mean(rewards[-args.patience_interval:])
            print(f"Episode {ep}\tReward {ep_r:.2f}\tAvg{args.patience_interval} {avg:.2f}")
            if avg > best_avg:
                best_avg = avg; no_improve = 0; agent.save(prefix="./")
            else:
                no_improve += 1
            frames = record_episode(agent, env)
            vid = f"temp_videos/ep{ep}.mp4"
            imageio.mimsave(vid, frames, fps=30)
            demo_video_paths.append(vid)

    plt.plot(rewards, label='Episode Reward')
    plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), label='Avg100')
    plt.legend()
    plt.savefig('learning_curve.png')

    writer = imageio.get_writer("training_progress.mp4", fps=30)
    for vp in demo_video_paths:
        reader = imageio.get_reader(vp)
        for frame in reader: writer.append_data(frame)
        reader.close()
    writer.close()
    shutil.rmtree("temp_videos")

    final_frames = record_episode(agent, env)
    imageio.mimsave("final_demo.mp4", final_frames, fps=30)
    print("Saved final_demo.mp4")