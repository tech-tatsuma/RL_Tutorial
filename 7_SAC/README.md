# Soft Actor-Critic (SAC) のアルゴリズム

このドキュメントは、Soft Actor-Critic (SAC) の仕組みと解説したものです。

## 概要

Soft Actor-Critic (SAC) は、最大エントロピー強化学習（Maximum Entropy Reinforcement Learning）に基づいたオフポリシー型の手法です。行動方策の学習に加えて、行動のランダム性（エントロピー）も最大化することで、安定かつ効率的な探索を行います。

## ネットワーク構成

* **Actor（方策ネットワーク）**: 状態 $s$ を入力として、正規分布の平均 $\mu$ と標準偏差 $\sigma$ を出力し、サンプリング後に $\tanh$ を通じて行動 $a$ を決定します。
* **Critic（状態価値関数）**: 状態 $s$ に対して状態価値 $V(s)$ を出力します。
* **Qネットワーク**: 状態 $s$ と行動 $a$ を入力として、$`Q(s, a)`$ を出力します。
* **Target Value Network**: Critic のスローポリシーコピーであり、安定したターゲット値を生成するために使用されます。

## 損失関数
### Q損失
```math
\mathcal{L}_{Q}
= \mathbb{E}\Bigl[\bigl(Q(s,a) - \bigl(r + \gamma (1 - d)\,V_{\mathrm{target}}(s')\bigr)\bigr)^{2}\Bigr]
```

### Value損失
```math
\mathcal{L}_{V}
= \mathbb{E}\Bigl[\bigl(V(s) - \bigl(Q(s,a') - \log \pi(a' \mid s)\bigr)\bigr)^{2}\Bigr]
```

### Policy損失
```math
\mathcal{L}_{\pi}
= \mathbb{E}\bigl[\log \pi(a \mid s) - Q(s,a)\bigr]
```

## エージェントの処理の流れ

1. 環境から状態 $s$ を観測し、方策ネットワークから行動 $a$ をサンプリング
2. 環境に行動 $a$ を実行し、報酬 $r$ と次の状態 $s'$ を取得
3. `(s, a, r, s', done)` をリプレイバッファに格納
4. 一定ステップごとにサンプルをバッチで取り出し、ネットワークを更新
5. ネットワークの更新時には、Actor・Critic・Qネットワークすべてを損失関数に基づいて同時に更新
6. ターゲットネットワークはソフトアップデート（ポリシーの移動平均）を行う

## 探索戦略

行動のサンプリングは、$`\mu`$ と $\sigma$ を用いた正規分布から行われ、$`\tanh`$ により動作範囲を制限します。加えてエントロピー項を損失関数に組み込むことで、過剰な決定性を避けた学習が可能です。