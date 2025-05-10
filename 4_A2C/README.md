# A2C (Advantage Actor-Critic) アルゴリズムの概要

## 概要

A2C（Advantage Actor-Critic）は、強化学習におけるポリシー勾配法の一種であり、
価値関数と方策を同時に学習する**アクター・クリティック法**に分類されます。

従来のActor-Criticとの違いは、同期的に複数環境で経験を収集して一括で更新する点です。
これにより、学習の安定性と効率が向上します。

---

## 構成要素

### 1. **Actor（ポリシーネットワーク）**

* 状態 \$s\$ に対して、確率的に行動 \$a\$ を選択する方策 \$\pi\_\theta(a|s)\$ を出力します。
* ネットワークの出力はSoftmaxを通じた確率分布です。

### 2. **Critic（価値関数ネットワーク）**

* 同じ入力状態 \$s\$ に対して、状態価値 \$V(s)\$ を推定します。
* この価値により、Actorの出力が良いか悪いかを評価します。

---

## 学習手順

1. 複数の環境（並列環境）で、\$n\$ ステップ分の経験を収集：

   * 状態 \$s\_t\$
   * 行動 \$a\_t\$
   * 報酬 \$r\_t\$
   * Done フラグ
2. 各ステップでの **Advantage** を計算：

$$
A_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

3. **Actor の損失**：

$$
\mathcal{L}_{\text{actor}} = -\log \pi_\theta(a_t|s_t) \cdot A_t
$$

4. **Critic の損失**：

$$
\mathcal{L}_{\text{critic}} = (V(s_t) - R_t)^2
$$

5. **エントロピー正則化**（探索を促すため）：

$$
\mathcal{L}_{\text{entropy}} = -\sum \pi(a|s) \log \pi(a|s)
$$

6. **総合損失関数**：

$$
\mathcal{L} = \mathcal{L}_{\text{actor}} + \lambda \cdot \mathcal{L}_{\text{critic}} + \beta \cdot \mathcal{L}_{\text{entropy}}
$$

---

## 本実装の特徴（コードの観点から）

* `SubprocVecEnv` によって複数環境を並列実行し、効率的に経験収集を行う。
* `torch.distributions.Categorical` を用いて確率分布から行動をサンプリング。
* `record_episode()` 関数によりエピソードをレンダリングして可視化・動画保存。
* Early stopping やベストモデル保存など、学習制御も実装済み。

---

## メリット

* **高いサンプル効率**（バッチで学習）
* **並列実行**により高速化
* **安定性向上**（バイアス・分散のトレードオフをうまく処理）

## デメリット

* 並列環境の構築が複雑（`SubprocVecEnv`など）
* 長期的な依存関係は捉えづらい（将来的にはLSTMなどの導入も検討）
