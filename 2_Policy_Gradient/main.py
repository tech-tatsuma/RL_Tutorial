import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imageio
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import os
import shutil
from itertools import count

# ============================
# Policy Network 定義
# ============================
class Policy(nn.Module):
    def __init__(self, num_state, num_action):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_state, 128)
        self.affine2 = nn.Linear(128, num_action)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

# ============================
# 動画記録用ヘルパー
# ============================
def record_episode(policy, env, max_steps=10000):
    frames = []
    state, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        # 決定的に最良行動を選択
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            probs = policy(state_t)
        action = probs.argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame = env.render()
        frames.append(frame)
        state = next_state
        steps += 1
    return frames, steps

# ============================
# エピソード完了時の更新処理
# ============================
def finish_episode(policy, optimizer, gamma, eps):
    R = 0
    policy_loss = []
    rewards = []
    # 割引リターンを逆順で計算
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    # 正規化
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    # ポリシー勾配損失
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    loss = torch.cat(policy_loss).sum()
    loss.backward()
    optimizer.step()
    # バッファクリア
    del policy.rewards[:]
    del policy.saved_log_probs[:]

# ============================
# メイン
# ============================
def main():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE + ログ + 可視化')
    parser.add_argument('--env',        type=str,   default='CartPole-v0')
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--seed',       type=int,   default=543)
    parser.add_argument('--max_eps',    type=int,   default=5000)
    parser.add_argument('--patience',   type=int,   default=200)
    parser.add_argument('--reward_win', type=float, default=195.0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_prefix',  type=str, default='pg')
    parser.add_argument('--fps',          type=int, default=30)
    args = parser.parse_args()

    # 環境の初期化
    env = gym.make(args.env, render_mode='rgb_array')

    num_state  = env.observation_space.shape[0]
    num_action = env.action_space.n

    # Policy, Optimizer, Logger, 各種バッファ
    policy    = Policy(num_state, num_action)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()
    rewards = []
    best_avg = -np.inf
    no_improve = 0

    # 動画保存用ディレクトリ準備
    os.makedirs("temp_videos", exist_ok=True)
    demo_video_paths = []

    # 学習ループ
    for i_episode in count(1):
        state, _ = env.reset()
        for t in range(1, 10001):
            # 行動選択＆保存
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            probs = policy(state_t)
            m = Categorical(probs)
            action = m.sample()
            action_int = action.item()
            policy.saved_log_probs.append(m.log_prob(action))
            next_state, reward, terminated, truncated, _ = env.step(action_int)
            done = terminated or truncated
            policy.rewards.append(reward)
            if done:
                break

        # エピソード長を記録
        rewards.append(t)

        # ポリシー更新
        finish_episode(policy, optimizer, args.gamma, eps)

        # 移動平均と early stopping 判定
        if i_episode >= 100:
            avg100 = np.mean(rewards[-100:])
            # 動画記録
            if i_episode % 100 == 0:
                frames, steps = record_episode(policy, env)
                video_path = f"temp_videos/demo_ep{i_episode}.mp4"
                imageio.mimsave(video_path, frames, fps=args.fps)
                demo_video_paths.append(video_path)
                print(f"[Video] episode {i_episode}, steps={steps}, saved to {video_path}")
            # モデル保存/早期終了
            if avg100 > best_avg:
                best_avg = avg100
                no_improve = 0
                best_model = f"{args.save_prefix}_best.pth"
                torch.save(policy.state_dict(), best_model)
                print(f"[Model] New best avg100={best_avg:.2f}, saved to {best_model}")
            else:
                no_improve += 1

        # ログ出力
        if i_episode % args.log_interval == 0:
            print(f"Episode {i_episode}\tLast length: {t}\tAvg100: {np.mean(rewards[-100:]):.2f}")

        # 早期終了条件
        if best_avg >= args.reward_win:
            print(f"Solved! avg100={best_avg:.2f} at episode {i_episode}")
            break
        if no_improve >= args.patience:
            print(f"Early stopping at episode {i_episode}, no improvement in {args.patience} eps")
            break
        if i_episode >= args.max_eps:
            print("Reached max episodes")
            break

    # 学習曲線を保存
    plt.figure()
    plt.plot(rewards, label='episode length')
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), label='avg100')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.legend()
    plt.savefig('learning_curve.png')
    print("Saved learning_curve.png")

    # 動画を一本に結合
    output_path = "training_progress.mp4"
    writer_vid = imageio.get_writer(output_path, fps=args.fps)
    for vp in demo_video_paths:
        reader = imageio.get_reader(vp)
        for frame in reader:
            writer_vid.append_data(frame)
        reader.close()
    writer_vid.close()
    print(f"Saved combined video to {output_path}")
    shutil.rmtree("temp_videos")

    # 最終デモ動画保存
    final_frames, final_steps = record_episode(policy, env)
    imageio.mimsave('final_demo.mp4', final_frames, fps=args.fps)
    print(f"Saved final_demo.mp4 (steps={final_steps})")

if __name__ == '__main__':
    main()