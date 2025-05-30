# Deep Q-Network (DQN) 

## アルゴリズム概要：DQN (Deep Q-Network)

DQN は、Q-Learning にニューラルネットワークを組み合わせたオフポリシー型の強化学習アルゴリズムで、離散的な行動空間に対して有効です。  
主な流れは以下のとおりです：

1. **Q関数の近似**  
   Q関数をニューラルネットワーク（`Net` クラス）で近似。

2. **経験再生（Experience Replay）**  
   エージェントの経験（遷移）をリプレイバッファに蓄積し、ミニバッチでランダムにサンプリングして学習に使用。

3. **ターゲットネットワーク**  
   安定化のため、学習対象とは別に固定されたターゲットネットワークを用いて Q 値のターゲットを計算。

4. **ε-greedy 方策**  
   行動選択時にランダム行動を混ぜることで、探索と活用のバランスを取る。

5. **TD誤差で学習**  
   時間差分誤差に基づく損失（MSE）を使って Q 値を更新。

## 学習アルゴリズムのステップ（Pseudocode）

1. `state = env.reset()`
2. `for episode in range(max_episodes):`
   - 行動 `a` を ε-greedy によって選択
   - 環境から `next_state`, `reward`, `done` を取得
   - 経験をメモリに保存
   - ミニバッチをサンプリングし、以下を計算:
     - TDターゲット: `reward + gamma * max(Q_target(next_state))`
     - TD誤差損失: `MSE(Q(state, action), TDターゲット)`
   - パラメータを勾配降下法で更新
   - ターゲットネットワークを定期的に同期

## 精度向上のための Tips / テクニック

- **経験リプレイ（Experience Replay）**
  - 時系列の相関を減らし、学習の安定性を向上
  - 容量制限付きのメモリに状態遷移を保存し、ランダムサンプリング

- **ターゲットネットワーク**
  - 学習対象ネットと分離することで学習が安定
  - 一定ステップごとに `act_net` の重みを `target_net` にコピー

- **εスケジューリング**
  - 初期は探索重視（ε=1.0）、学習が進むと徐々に ε を減らして活用を強める（ε=0.01）

- **Early Stopping**
  - 直近100エピソードの平均報酬が改善しない場合に学習を終了