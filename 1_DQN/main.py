import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import imageio
import matplotlib.pyplot as plt
from collections import namedtuple
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import shutil
import os

# ============================
# Qネットワークの定義（2層の全結合ニューラルネット）
# ============================
class Net(nn.Module):
    def __init__(self, num_state, num_action):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ============================
# DQNエージェントの定義（行動ネットワークとターゲットネットワークを持つ）
# ============================
class DQN:
    def __init__(self, num_state, num_action, capacity=8000, lr=1e-3, gamma=0.995,
                 batch_size=256, update_target_every=100):
        self.act_net    = Net(num_state, num_action) # 学習対象のネットワーク
        self.target_net = Net(num_state, num_action) # ターゲットネットワーク
        self.target_net.load_state_dict(self.act_net.state_dict()) # 初期状態をコピー
        self.memory     = [] # 経験リプレイメモリ
        self.capacity   = capacity # メモリ上限
        self.gamma      = gamma # 割引率
        self.batch_size = batch_size # バッチサイズ
        self.optimizer  = torch.optim.Adam(self.act_net.parameters(), lr=lr) # 最適化関数
        self.loss_func  = nn.MSELoss() # 損失関数
        self.update_count = 0 # 更新回数
        self.update_target_every = update_target_every # ターゲットネット更新感覚

    def select_action(self, state, epsilon=0.1):
        """ε-greedyにより行動選択"""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            qvals = self.act_net(state_t)
        if np.random.rand() < epsilon:
            return np.random.randint(qvals.size(1))
        return qvals.argmax(dim=1).item()

    def store(self, transition):
        """経験をメモリに保存"""
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)

    def update(self):
        # 学習バッチが溜まっていなければ更新しない
        if len(self.memory) < self.batch_size:
            return

        # リプレイバッファからランダムにバッチデータを取得
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.tensor([self.memory[i].state for i in batch], dtype=torch.float32)
        actions= torch.tensor([self.memory[i].action for i in batch], dtype=torch.int64).view(-1,1)
        rewards = torch.tensor([self.memory[i].reward for i in batch], dtype=torch.float32)
        next_states = torch.tensor([self.memory[i].next_state for i in batch], dtype=torch.float32)
        dones = torch.tensor([float(self.memory[i].done)  for i in batch], dtype=torch.float32)

        # ターゲットQ値の計算
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            next_q = next_q * (1.0 - dones)
            target = rewards + self.gamma * next_q

        # 現在のQ値予測を取得
        pred = self.act_net(states).gather(1, actions).squeeze()
        loss = self.loss_func(pred, target) # MSE損失の計算

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ターゲットネットを一定間隔で更新
        self.update_count += 1
        if self.update_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.act_net.state_dict())

# ============================
# モデルの動作を記録し，動画として保存
# ============================
def record_episode(model, env, max_steps=10000):
    frames = []
    state, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = model.select_action(state, epsilon=0.0)
        next_state, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        state = next_state
        done = terminated or truncated
        steps += 1
    return frames, steps

# ============================
# 学習ループと早期終了処理
# ============================
def main():
    # 入力引数を処理
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',       type=str, default='CartPole-v1')
    parser.add_argument('--max_eps',   type=int, default=5000)
    parser.add_argument('--patience',  type=int, default=200)
    parser.add_argument('--reward_win',type=float, default=195.0)
    parser.add_argument('--save_prefix', type=str, default='dqn')
    parser.add_argument('--fps',       type=int, default=30)
    args = parser.parse_args()

    # 実験環境の準備
    env = gym.make(args.env, render_mode='rgb_array')
    num_state  = env.observation_space.shape[0]
    num_action = env.action_space.n

    agent = DQN(num_state, num_action) # Agentの初期化
    Transition = namedtuple('Transition', ['state','action','reward','next_state','done']) # リプレイバッファの初期化

    # εのスケジューリング
    epsilon_start, epsilon_end, eps_decay = 1.0, 0.01, 1000

    best_avg = -np.inf
    best_model_path = None
    no_improve = 0
    rewards = []
    os.makedirs("temp_videos", exist_ok=True)
    demo_video_paths = []
    
    for ep in range(1, args.max_eps+1):
        state, _ = env.reset() # 環境をリセット
        total_r = 0.0
        done = False
        epsilon = max(epsilon_end, epsilon_start - ep*(epsilon_start-epsilon_end)/eps_decay) # epsilonのスケジューリング
        while not done:
            action = agent.select_action(state, epsilon) # 行動を決定
            next_state, r, terminated, truncated, _ = env.step(action) # 行動を実行し、環境からのフィードバックを取得
            done = terminated or truncated
            agent.store(Transition(state, action, r, next_state, done)) # リプレイバッファに格納
            agent.update() # ネットワークを学習
            state = next_state
            total_r += r
        rewards.append(total_r)
        
        # 移動平均で性能を評価
        if ep >= 100:
            avg100 = np.mean(rewards[-100:])
            frames, steps = record_episode(agent, env)
            video_path = f"temp_videos/demo_ep{ep}.mp4"
            imageio.mimsave(video_path, frames, fps=args.fps)
            demo_video_paths.append(video_path)
            print(f"Saved demo at episode {ep} (steps={steps}) to {video_path}")
            
            # モデル保存：ベスト更新時
            if avg100 > best_avg:
                best_avg = avg100
                no_improve = 0
                best_model_path = f"{args.save_prefix}_best.pth"
                torch.save(agent.act_net.state_dict(), best_model_path)
            else:
                no_improve += 1
        
        # 早期終了条件
        if best_avg >= args.reward_win:
            print(f"Solved! Avg100={best_avg:.2f} at ep={ep}")
            break
        if no_improve >= args.patience:
            print(f"Early stopping at ep={ep}, no improvement in {args.patience} eps")
            break
        if ep % 500 == 0:
            torch.save(agent.act_net.state_dict(), f"{args.save_prefix}_ep{ep}.pth")

    print(f"Best avg100: {best_avg:.2f}, model: {best_model_path}")

    # 学習曲線を保存
    plt.figure()
    plt.plot(rewards, label='episode reward')
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), label='avg100')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('learning_curve.png')
    print('Saved learning curve to learning_curve.png')

    output_path = "training_progress.mp4"
    writer = imageio.get_writer(output_path, fps=args.fps)

    for video_path in demo_video_paths:
        reader = imageio.get_reader(video_path)
        for frame in reader:
            writer.append_data(frame)
        reader.close()

    writer.close()
    print(f"Saved combined training video to {output_path}")

    shutil.rmtree("temp_videos")

    # 学習後のモデルで動画を保存
    frames, steps = record_episode(agent, env)
    imageio.mimsave('final_demo.mp4', frames, fps=args.fps)
    print(f"Saved final demo (steps={steps}) to final_demo.mp4")

if __name__ == '__main__':
    main()
