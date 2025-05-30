# Q学習 (Q-Learning) アルゴリズムの解説

このドキュメントは、 **Q学習 (Q-Learning)** の基本的な仕組みを実装方法とともに紹介します。

---

## Q学習とは？

Q学習は、**モデルフリーな強化学習アルゴリズム**で、エージェントがある環境内で行動を選択しながら、最も多くの報酬を得るような方策（policy）を学習します。

Q値（行動価値関数）を更新しながら、最適な行動選択ルール（方策）を学習していきます。

---

## Q値更新の数式

Q学習では以下の **ベルマン方程式に基づく更新則** に従ってQ値を更新します。

```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \cdot \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
```

- $`Q(s_t, a_t)`$ ：状態 $`s_t`$ において行動 $`a_t`$ を選択したときのQ値
- $`\alpha`$：学習率（Learning rate）
- $`r_t`$：その行動によって得られる報酬
- $`\gamma`$：割引率（Discount factor）、将来の報酬の重要度
- $`\max_{a'} Q(s_{t+1}, a')`$：次の状態における最大のQ値（最適行動の価値）

---

## アルゴリズムのステップ

1. Qテーブル（状態×行動の表）を初期化する（すべて0）
2. 各エピソードに対して次を繰り返す：
    - 初期状態 $`s_0`$ を設定
    - 終端状態（ゴール）に到達するまで以下を繰り返す：
        1. ε-greedy 方策で行動 $`a_t`$ を選ぶ：
            - ランダム：確率 $`\epsilon`$
            - 最大Q値の行動：確率 $`1-\epsilon`$
        2. 行動により次の状態 $`s_{t+1}`$ と報酬 $`r_t`$ を得る
        3. Q値を更新する（上記の更新式）
        4. 状態を $`s_{t+1}`$ に更新
3. 学習が完了したら、Qテーブルを用いて最適方策を得る

---

## ε-greedy 方策

Q学習では **探索 (exploration)** と **活用 (exploitation)** のバランスを取る必要があります。

```math
\pi(s) =
\begin{cases}
\text{ランダムな行動} & \text{確率 } \epsilon \\
\arg\max_a Q(s, a) & \text{確率 } 1 - \epsilon
\end{cases}
```

探索により未知の行動を試し、活用によって既知の良い行動を強化します。

---

## 本プログラムの環境設定

- 状態空間：線形の20マス（`N_STATE = 20`）
- 行動空間：`left`, `right`
- ゴール：19番目（右端）にある「T」に到達すると終了
- 移動時の報酬：`-0.5`
- ゴール報酬：`+1`
- 学習回数：`MAX_EPISODES = 200`

---

## パラメータ設定

| パラメータ名 | 意味 | 値 |
|--------------|------|----|
| `ALPHA`      | 学習率 $`\alpha`$ | 0.1 |
| `GAMMA`      | 割引率 $`\gamma`$ | 0.95 |
| `EPSILION`   | ε-greedy法のε       | 0.9 |
| `MAX_EPISODES` | エピソード数     | 200 |

---

## 参考文献

- [R. Sutton and A. Barto, "Reinforcement Learning: An Introduction"](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- [OpenAI Spinning Up Documentation](https://spinningup.openai.com/)