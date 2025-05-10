import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from gridworld.main import GridWorld

ALPHA = 0.1           # 学習率
GAMMA = 0.95          # 割引率
EPSILION = 0.9        # ε-greedy の確率
MAX_EPISODES = 50     # 総エピソード数
FRESH_TIME = 0.1      # 表示更新の間隔（秒）
ACTIONS = [0, 1, 2, 3]  # 0:UP, 1:RIGHT, 2:DOWN, 3:LEFT

# 状態数 = 行数 x 列数 ではなく、座標 (row, col) をインデックスにする
def build_q_table(rows, cols, actions):
    index = pd.MultiIndex.from_product([range(rows), range(cols)], names=["row", "col"])
    return pd.DataFrame(0.0, index=index, columns=actions)

def choose_action(state, q_table):
    """ε-greedy方策で行動選択"""
    state_action = q_table.loc[tuple(state), :]
    if np.random.uniform() > EPSILION or (state_action == 0).all():
        return np.random.choice(ACTIONS)
    else:
        return state_action.idxmax()

def sarsa_learning(env):
    q_table = build_q_table(env.world_row, env.world_col, ACTIONS)
    step_counter_times = []

    for episode in range(MAX_EPISODES):
        state = env.reset(exploring_starts=False)  # 左下から開始
        action = choose_action(state, q_table)
        done = False
        step_counter = 0
        env.render()
        time.sleep(FRESH_TIME)

        while not done:
            next_state, reward, done = env.step(action)
            if not done:
                next_action = choose_action(next_state, q_table)
                q_target = reward + GAMMA * q_table.loc[tuple(next_state), next_action]
            else:
                q_target = reward
                next_action = None

            q_predict = q_table.loc[tuple(state), action]
            q_table.loc[tuple(state), action] += ALPHA * (q_target - q_predict)

            state = next_state
            action = next_action
            step_counter += 1

            env.render()
            time.sleep(FRESH_TIME)

        print(f"Episode {episode+1}, steps: {step_counter}")
        step_counter_times.append(step_counter)

    return q_table, step_counter_times

def main():
    # 環境構築
    env = GridWorld(3, 4)
    state_matrix = np.array([
        [0,  0,  0, +1],
        [0, -1,  0, -1],
        [0,  0,  0, +1]
    ])
    reward_matrix = np.zeros((3, 4))
    reward_matrix[0, 3] = 1
    reward_matrix[2, 3] = 1
    transition_matrix = np.array([
        [0.8, 0.1, 0.05, 0.05],  # UP
        [0.1, 0.8, 0.05, 0.05],  # RIGHT
        [0.05, 0.1, 0.8, 0.05],  # DOWN
        [0.05, 0.05, 0.1, 0.8],  # LEFT
    ])

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    # 学習実行
    q_table, step_counter_times = sarsa_learning(env)

    # 学習結果の表示
    print("学習後のQテーブル：\n", q_table)
    plt.plot(step_counter_times, 'b-')
    plt.xlabel("Episode")
    plt.ylabel("Steps to goal")
    plt.title("SARSA in GridWorld")
    plt.show()

main()