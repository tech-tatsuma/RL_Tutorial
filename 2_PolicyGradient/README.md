# REINFORCEアルゴリズムによる強化学習

ここでは、**REINFORCE アルゴリズム** を使用して、`CartPole-v0` 環境における最適なポリシー（方策）を学習します。

## アルゴリズム概要

REINFORCEは、エピソード単位で状態・行動・報酬の系列を収集し、行動方策のパラメータを確率的勾配上昇法によって更新します。主な手順は以下の通りです：

1. エピソードをサンプルする
2. 各タイムステップでログ確率と報酬を記録する
3. 各ステップの **割引累積報酬（リターン）** を計算する
4. ログ確率 × リターンに基づいて損失を定義し、逆伝播でパラメータを更新する

## 損失関数

各タイムステップ $`t`$ における損失関数は以下の通りです：

```math
L(\theta) = - \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) \cdot R_t
```

ここで $`R_t`$ は時刻 $`t`$ からの割引累積報酬、$`\pi_{\theta}`$ は方策ネットワークです。

## 割引累積報酬の計算

プログラムでは、報酬系列 $`r_t`$ を逆順にたどって以下のようにリターン $`R_t`$ を計算します：
```
R = 0
returns = []
for r in reversed(policy.rewards):
R = r + gamma * R
returns.insert(0, R)
```
## 正規化

安定性向上のため、リターンは以下のようにゼロ平均・単位分散に正規化されます：
```
returns = torch.tensor(returns)
returns = (returns - returns.mean()) / (returns.std() + eps)
```
## 損失の最終計算と更新

行動ログ確率と正規化されたリターンを組み合わせて損失を計算し、勾配を逆伝播します：
```
policy_loss = []
for log_prob, reward in zip(policy.saved_log_probs, returns):
policy_loss.append(-log_prob * reward)
loss = torch.cat(policy_loss).sum()
loss.backward()
optimizer.step()
```