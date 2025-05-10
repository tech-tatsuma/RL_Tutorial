import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import imageio

# ==== Qネットワーク ====
class Net(nn.Module):
    def __init__(self, num_state, num_action):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ==== 行動選択関数 ====
def select_action(model, state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)
    return q_values.argmax(dim=1).item()

# ==== 動画記録関数 ====
def record_episode(model, env, output_path='demo.mp4', fps=30, max_steps=10000):
    frames = []
    state, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = select_action(model, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame = env.render()
        frames.append(frame)
        state = next_state
        steps += 1

    imageio.mimsave(output_path, frames, fps=fps)
    print(f"🎬 動画を保存しました: {output_path}（{steps}ステップ）")

# ==== メイン関数 ====
def main(args):
    # 環境作成（RGBフレーム取得用）
    env = gym.make('CartPole-v0', render_mode='rgb_array')
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n

    # モデルの作成と重み読み込み
    model = Net(num_state, num_action)
    model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    model.eval()

    # 動画の生成
    record_episode(model, env, output_path=args.output, fps=args.fps)

# ==== 引数パーサ ====
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="重みファイルを使ってCartPoleの動作を録画")
    parser.add_argument('--weight_path', type=str, required=True, help='重みファイル（.pth）へのパス')
    parser.add_argument('--output', type=str, default='cartpole_demo.mp4', help='出力動画のファイル名')
    parser.add_argument('--fps', type=int, default=30, help='動画のFPS（フレーム/秒）')
    args = parser.parse_args()
    main(args)