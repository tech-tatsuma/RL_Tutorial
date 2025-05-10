import argparse
import os
import gym
import shutil
import imageio
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# ============================
# モデル定義 (Actor-Critic)
# ============================
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self, num_state, num_action):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.action_head = nn.Linear(128, num_action)
        self.value_head = nn.Linear(128, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.action_head(x), dim=-1), self.value_head(x)

# ============================
# 動画記録関数
# ============================
def record_episode(policy, env, max_steps=10000):
    frames = []
    state, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs, _ = policy(state_t)
        action = probs.argmax(dim=1).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        state = next_state
        done = terminated or truncated
        steps += 1
    return frames, steps

# ============================
# エピソード終了後の更新処理
# ============================
def finish_episode(policy, optimizer, gamma, eps):
    R = 0
    saved_actions = policy.saved_actions
    policy_loss = []
    value_loss = []
    returns = []

    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), r in zip(saved_actions, returns):
        advantage = r - value.item()
        policy_loss.append(-log_prob * advantage)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.saved_actions[:]

# ============================
# メイン関数
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--max_eps', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--reward_win', type=float, default=195.0)
    parser.add_argument('--save_prefix', type=str, default='ac')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    env = gym.make(args.env, render_mode='rgb_array')
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n

    policy = Policy(num_state, num_action)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()
    rewards = []
    best_avg = -np.inf
    no_improve = 0
    os.makedirs("temp_videos", exist_ok=True)
    demo_video_paths = []

    for ep in range(1, args.max_eps + 1):
        state, _ = env.reset()
        total_r = 0
        for t in range(1, 10001):
            state_t = torch.tensor(state, dtype=torch.float32)
            probs, value = policy(state_t)
            m = Categorical(probs)
            action = m.sample()
            policy.saved_actions.append(SavedAction(m.log_prob(action), value))
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            policy.rewards.append(reward)
            total_r += reward
            state = next_state
            if terminated or truncated:
                break

        rewards.append(total_r)
        finish_episode(policy, optimizer, gamma=0.99, eps=eps)

        if ep >= 100:
            avg100 = np.mean(rewards[-100:])
            if ep % 100 == 0:
                frames, steps = record_episode(policy, env)
                video_path = f"temp_videos/demo_ep{ep}.mp4"
                imageio.mimsave(video_path, frames, fps=args.fps)
                demo_video_paths.append(video_path)

            if avg100 > best_avg:
                best_avg = avg100
                no_improve = 0
                best_model_path = f"{args.save_prefix}_best.pth"
                torch.save(policy.state_dict(), best_model_path)
                print(f"[Model] Saved best model with avg100={avg100:.2f} at ep={ep}")
            else:
                no_improve += 1

        if ep % 10 == 0:
            print(f"Episode {ep}\tReward: {total_r}\tAvg100: {np.mean(rewards[-100:]):.2f}")

        if best_avg >= args.reward_win:
            print(f"Solved! Avg100={best_avg:.2f} at ep={ep}")
            break
        if no_improve >= args.patience:
            print(f"Early stopping at ep={ep} (no improvement in {args.patience} eps)")
            break

    # 学習曲線
    plt.figure()
    plt.plot(rewards, label='episode reward')
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), label='avg100')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('learning_curve.png')

    # 動画を1本に結合
    output_path = "training_progress.mp4"
    writer = imageio.get_writer(output_path, fps=args.fps)
    for vp in demo_video_paths:
        reader = imageio.get_reader(vp)
        for frame in reader:
            writer.append_data(frame)
        reader.close()
    writer.close()
    shutil.rmtree("temp_videos")

    # 最終デモ
    final_frames, final_steps = record_episode(policy, env)
    imageio.mimsave("final_demo.mp4", final_frames, fps=args.fps)
    print(f"Saved final_demo.mp4 (steps={final_steps})")

if __name__ == '__main__':
    main()
