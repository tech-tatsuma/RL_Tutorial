import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import namedtuple
import matplotlib.pyplot as plt
import imageio
import shutil
import gym

# ================================
# ハイパーパラメータ
# ================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Pendulum-v1")
parser.add_argument('--tau', default=0.005, type=float)
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--gradient_steps', default=1, type=int)
parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--capacity', default=10000, type=int)
parser.add_argument('--iteration', default=100000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--log_interval', default=50, type=int)
parser.add_argument('--load', default=False, type=bool)
args = parser.parse_args()

# ================================
# 環境とリプレイバッファの定義
# ================================
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])
env = gym.make(args.env_name, render_mode='rgb_array')
env.reset(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# 状態，行動の次元取得
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float()

# ================================
# Actor Network
# ================================
class Actor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x), -20, 2)
        return mu, log_std


# ================================
# Critic Network
# ================================
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


# ================================
# 状態行動価値 Network
# ================================
class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)

# ================================
# SAC Agent
# ================================
class SAC:
    def __init__(self):
        # ネットワークの初期化
        self.policy_net = Actor(state_dim).to(device)
        self.value_net = Critic(state_dim).to(device)
        self.Q_net = Q(state_dim, action_dim).to(device)
        self.target_value_net = Critic(state_dim).to(device)

        # オプティマイザの初期化
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.learning_rate)
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=args.learning_rate)

        # リプレイバッファの初期化
        self.replay_buffer = []
        self.num_transition = 0
        self.value_criterion = nn.MSELoss()
        self.Q_criterion = nn.MSELoss()
        self.capacity = args.capacity

        # ターゲットネットワークの同期
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        os.makedirs('SAC_model', exist_ok=True)

    def select_action(self, state):
        """
        正規分布に基づく確率的な行動選択
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mu, log_std = self.policy_net(state)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.sample()
        action = torch.tanh(z)
        return action.cpu().detach().numpy()[0]

    def store(self, s, a, r, s_, d):
        """
        リプレイバッファに保存
        """
        if len(self.replay_buffer) < self.capacity:
            self.replay_buffer.append(Transition(s, a, r, s_, d))
        else:
            idx = self.num_transition % self.capacity
            self.replay_buffer[idx] = Transition(s, a, r, s_, d)
        self.num_transition += 1

    def update(self):
        """
        一定以上のデータが溜まったら学習開始
        """
        if self.num_transition < args.capacity:
            return

        indices = np.random.choice(len(self.replay_buffer), args.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        s = torch.FloatTensor([b.s for b in batch]).to(device)
        a = torch.FloatTensor([b.a for b in batch]).to(device)
        r = torch.FloatTensor([b.r for b in batch]).unsqueeze(1).to(device)
        s_ = torch.FloatTensor([b.s_ for b in batch]).to(device)
        d = torch.FloatTensor([b.d for b in batch]).unsqueeze(1).to(device)

        # Q関数の更新
        with torch.no_grad():
            target_v = self.target_value_net(s_)
            target_q = r + (1 - d) * args.gamma * target_v

        expected_q = self.Q_net(s, a)
        q_loss = self.Q_criterion(expected_q, target_q)

        mu, log_std = self.policy_net(s)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + min_Val)
        log_prob = log_prob.sum(1, keepdim=True)

        q_pi = self.Q_net(s, action)
        value_target = q_pi - log_prob
        value = self.value_net(s)
        v_loss = self.value_criterion(value, value_target.detach())

        policy_loss = (log_prob - q_pi).mean()

        self.Q_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        total_loss = q_loss + v_loss + policy_loss
        total_loss.backward()
        self.Q_optimizer.step()
        self.value_optimizer.step()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1 - args.tau) + param.data * args.tau)

    def save(self):
        torch.save(self.policy_net.state_dict(), 'SAC_model/policy.pth')

# ================================
# エピソードの録画（動画用）
# ================================
def record_episode(agent, env, max_steps=200, fps=30):
    frames = []
    state, _ = env.reset()
    for _ in range(max_steps):
        action = agent.select_action(state)
        state, _, done, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        if done or truncated:
            break
    return frames

# ================================
# Training loop
# ================================
def main():
    agent = SAC()
    rewards, demo_video_paths = [], []
    best_avg, no_improve = -np.inf, 0
    os.makedirs('temp_videos', exist_ok=True)

    for i in range(args.iteration):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(200):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.store(state, action, reward, next_state, done or truncated)
            agent.update()
            state = next_state
            ep_reward += reward
            if done or truncated:
                break

        rewards.append(ep_reward)

        if i % args.log_interval == 0:
            avg50 = np.mean(rewards[-50:])
            print(f"Episode {i}\tReward {ep_reward:.2f}\tAvg50 {avg50:.2f}")
            frames = record_episode(agent, env)
            path = f"temp_videos/ep{i}.mp4"
            imageio.mimsave(path, frames, fps=30)
            demo_video_paths.append(path)
            if avg50 > best_avg:
                best_avg = avg50
                no_improve = 0
                agent.save()
            else:
                no_improve += 1
            if best_avg >= -200.0 or no_improve >= 20:
                print("Early stopping.")
                break

    plt.plot(rewards, label='Episode Reward')
    plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), label='Avg50')
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

if __name__ == '__main__':
    main()
