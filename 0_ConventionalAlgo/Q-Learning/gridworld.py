import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from gridworld.main import GridWorld

ALPHA = 0.1
GAMMA = 0.95
EPSILION = 0.9
MAX_EPISODES = 50
FRESH_TIME = 0.1

# グリッドワールドの設定（縦3x横4）
ROWS, COLS = 3, 4
ACTIONS = [0, 1, 2, 3]  # 0:UP, 1:RIGHT, 2:DOWN, 3:LEFT

def build_q_table(rows, cols, actions):
    """状態=位置(row, col), 行動=ACTIONS。すべての組に対してQ値を持つ。"""
    index = pd.MultiIndex.from_product([range(rows), range(cols)], names=["row", "col"])
    q_table = pd.DataFrame(0.0, index=index, columns=actions)
    return q_table

def choose_action(state, q_table):
    """ε-greedy方策による行動選択"""
    state_action = q_table.loc[tuple(state), :]
    if np.random.uniform() > EPSILION or (state_action == 0).all():
        action = np.random.choice(ACTIONS)
    else:
        action = state_action.idxmax()
    return action

def q_learning(env):
    q_table = build_q_table(env.world_row, env.world_col, ACTIONS)
    step_counter_times = []

    for episode in range(MAX_EPISODES):
        state = env.reset(exploring_starts=False)
        is_terminal = False
        step_counter = 0
        env.render()
        time.sleep(FRESH_TIME)

        while not is_terminal:
            action = choose_action(state, q_table)
            next_state, reward, done = env.step(action)

            # Q値更新
            if done:
                q_target = reward
                is_terminal = True
            else:
                q_target = reward + GAMMA * q_table.loc[tuple(next_state), :].max()

            q_predict = q_table.loc[tuple(state), action]
            q_table.loc[tuple(state), action] += ALPHA * (q_target - q_predict)

            state = next_state
            env.render()
            time.sleep(FRESH_TIME)
            step_counter += 1

        print(f"Episode {episode+1}, steps: {step_counter}")
        step_counter_times.append(step_counter)

    return q_table, step_counter_times

def main():
    # GridWorld環境の構築
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

    # Q学習実行
    q_table, step_counter_times = q_learning(env)

    # 結果出力
    print("学習されたQテーブル：\n", q_table)
    plt.plot(step_counter_times, 'g-')
    plt.xlabel("Episode")
    plt.ylabel("Steps to goal")
    plt.title("Q-learning in GridWorld")
    plt.show()

main()