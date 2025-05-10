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
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from multiprocessing_env import SubprocVecEnv # 複数環境を並列に動かすためのユーティリティ

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# ============================
# Actor-Critic ネットワーク
# ============================
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()
        # 状態価値関数
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        # 方策
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value

# ============================
# 複数環境の生成用
# ============================
def make_env(env_name):
    def _thunk():
        return gym.make(env_name)
    return _thunk

# ============================
# テスト時に動画とステップ数を記録
# ============================
def record_episode(model, env, max_steps=10000):
    frames = []

    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    done = False
    steps = 0
    while not done and steps < max_steps:
        st = torch.from_numpy(state).float().unsqueeze(0)

        with torch.no_grad():
            dist, _ = model(st)
        action = dist.probs.argmax(dim=1).item()

        step_out = env.step(action)
        if len(step_out) == 5:
            next_state, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_out

        state = next_state
        frames.append(env.render())
        steps += 1
    return frames, steps

# ============================
# 割引報酬を計算
# ============================
def compute_returns(next_value, rewards, masks, gamma):
    R = next_value
    returns = []
    for r, m in zip(reversed(rewards), reversed(masks)):
        R = r + gamma * R * m
        returns.insert(0, R)
    return returns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',            type=str,   default='CartPole-v1')
    parser.add_argument('--num_envs',       type=int,   default=8)
    parser.add_argument('--hidden_size',    type=int,   default=256)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--num_steps',      type=int,   default=5)
    parser.add_argument('--max_frames',     type=int,   default=20000)
    parser.add_argument('--test_interval',  type=int,   default=1000)
    parser.add_argument('--patience',       type=int,   default=10)
    parser.add_argument('--reward_win',     type=float, default=195.0)
    parser.add_argument('--save_prefix',    type=str,   default='a2c')
    parser.add_argument('--fps',            type=int,   default=30)
    args = parser.parse_args()

    # 環境初期化
    envs = SubprocVecEnv([make_env(args.env) for _ in range(args.num_envs)])
    single_env = gym.make(args.env, render_mode='rgb_array')
    num_inputs  = envs.observation_space.shape[0]
    num_outputs = envs.action_space.n

    # モデルとオプティマイザの初期化
    model     = ActorCritic(num_inputs, num_outputs, args.hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    eps       = np.finfo(np.float32).eps.item()

    os.makedirs("temp_videos", exist_ok=True)
    demo_video_paths = []
    test_rewards = []
    best_avg     = -np.inf
    no_improve   = 0

    state = envs.reset()

    frame_idx = 0

    # 学習ループ
    while frame_idx < args.max_frames:
        log_probs, values, rewards, masks = [], [], [], []

        # num_steps 分だけ収集
        for _ in range(args.num_steps):
            st = torch.from_numpy(state).float()
            dist, value = model(st)
            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(torch.tensor(reward, dtype=torch.float32).unsqueeze(1))
            masks.append(torch.tensor(1 - done, dtype=torch.float32).unsqueeze(1))

            state = next_state
            frame_idx += 1

            # 定期的にテスト＆可視化
            if frame_idx % args.test_interval == 0:
                avg_r = np.mean([test_env(single_env, model) for _ in range(5)])
                test_rewards.append(avg_r)
                print(f"Frame {frame_idx}\tTest reward: {avg_r:.2f}")

                # 学習曲線保存
                plt.figure(); plt.plot(test_rewards); plt.savefig("learning_curve.png"); plt.close()

                # Early stopping 判定
                if avg_r > best_avg:
                    best_avg = avg_r; no_improve = 0
                    torch.save(model.state_dict(), f"{args.save_prefix}_best.pth")
                else:
                    no_improve += 1
                
                vid, steps = record_episode(model, single_env)
                path = f"temp_videos/demo_{frame_idx}.mp4"
                imageio.mimsave(path, vid, fps=args.fps)
                demo_video_paths.append(path)
                if avg_r >= args.reward_win or no_improve >= args.patience:
                    print("Early stopping triggered.")
                    frame_idx = args.max_frames  # ループ脱出
                    break

        # バックアップ値
        next_st = torch.from_numpy(state).float()
        _, next_value = model(next_st)
        returns = compute_returns(next_value, rewards, masks, gamma=0.99)

        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)

        advantage    = returns - values
        actor_loss   = -(log_probs * advantage.detach()).mean()
        critic_loss  = advantage.pow(2).mean()
        dist, _      = model(torch.from_numpy(state).float())
        entropy_loss = -dist.entropy().mean()

        loss = actor_loss + 0.5 * critic_loss + 0.001 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 総合動画作成
    writer = imageio.get_writer("training_progress.mp4", fps=args.fps)
    for vp in demo_video_paths:
        reader = imageio.get_reader(vp)
        for frame in reader:
            writer.append_data(frame)
        reader.close()
    writer.close()
    shutil.rmtree("temp_videos")

    # 最終デモ動画
    final_vid, final_steps = record_episode(model, single_env)
    imageio.mimsave("final_demo.mp4", final_vid, fps=args.fps)
    print(f"Saved final demo ({final_steps} steps)")

# ========================
# テスト用評価関数
# ========================
def test_env(env, model):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        state, _ = reset_out
    else:
        state = reset_out

    done = False
    total = 0
    while not done:
        st = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        with torch.no_grad():
            dist, _ = model(st)
        action = dist.probs.argmax(dim=1).item()

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, r, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            obs, r, done, _ = step_out

        state = obs
        total += r
    return total

if __name__ == '__main__':
    main()