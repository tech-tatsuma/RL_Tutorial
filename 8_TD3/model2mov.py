#!/usr/bin/env python3
# make_video.py

import argparse
import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio

# 1) Actor ネットワーク定義（学習スクリプトと同一構造）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.max_action

# 2) エピソードを通してフレームを取得する関数
def record_episode(actor, env, max_steps, device):
    frames = []
    state, _ = env.reset()
    for _ in range(max_steps):
        # 状態を Tensor に変換
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor(s).cpu().numpy().flatten()
        # 環境ステップ
        next_state, reward, done, truncated, _ = env.step(action)
        # 描画フレーム取得
        frame = env.render()
        frames.append(frame)
        state = next_state
        if done or truncated:
            break
    return frames

def main():
    parser = argparse.ArgumentParser(description="学習済みTD3モデルで動画生成")
    parser.add_argument("--env",      type=str,   default="Pendulum-v1", help="Gym 環境名")
    parser.add_argument("--model",    type=str,   required=True,         help="読み込む actor モデルのファイルパス（.pth）")
    parser.add_argument("--output",   type=str,   default="demo.mp4",    help="出力動画ファイル名")
    parser.add_argument("--episodes", type=int,   default=5,             help="動画を生成するエピソード数")
    parser.add_argument("--max_steps",type=int,   default=200,           help="1エピソードあたりの最大ステップ数")
    parser.add_argument("--fps",      type=int,   default=30,            help="動画のFPS")
    args = parser.parse_args()

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 環境準備
    env = gym.make(args.env, render_mode="rgb_array")

    # モデル定義＆読み込み
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    actor = Actor(state_dim, action_dim, max_action).to(device)
    actor.load_state_dict(torch.load(args.model, map_location=device))
    actor.eval()

    # 全エピソード通してフレーム収集
    all_frames = []
    for ep in range(args.episodes):
        frames = record_episode(actor, env, args.max_steps, device)
        print(f"Episode {ep+1}: collected {len(frames)} frames.")
        all_frames.extend(frames)

    # 出力ディレクトリ作成
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # 動画保存
    imageio.mimsave(args.output, all_frames, fps=args.fps)
    print(f"Saved video to {args.output}")

if __name__ == "__main__":
    main()