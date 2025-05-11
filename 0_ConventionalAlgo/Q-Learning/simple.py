import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

ALPHA = 0.1 # 学習率
GAMMA = 0.95 # 割引率
EPSILION = 0.9 # ε-greedy法におけるランダム行動の確率
N_STATE = 20 # 状態の数
ACTIONS = ['left', 'right'] # 可能な行動
MAX_EPISODES = 50 # エピソード数
FRESH_TIME = 0.1 # 表示更新のための待機時間

def build_q_table(n_state, actions):
    """
    Qテーブルの初期化（全ての状態-行動ペアに0を設定）
    """
    q_table = pd.DataFrame(
    np.zeros((n_state, len(actions))),
    np.arange(n_state), # 状態(行)
    actions # 行動(列)
    )
    return q_table

def choose_action(state, q_table):
    """
    行動選択（ε-greedy方策）
    """
    state_action = q_table.loc[state,:]
    if np.random.uniform()>EPSILION or (state_action==0).all():
        # ランダム行動
        action_name = np.random.choice(ACTIONS)
    else:
        # 最大Q値を持つ行動を選択
        action_name = state_action.idxmax()
    return action_name

def get_env_feedback(state, action):
    """
    環境からのフィードバック（状態遷移と報酬）
    """
    if action=='right':
        if state == N_STATE-2:
            next_state = 'terminal' # 終端状態に到達
            reward = 1 # ゴール報酬
        else:
            next_state = state+1 # 右に移動
            reward = -0.5 # 移動ペナルティ
    else:
        if state == 0:
            next_state = 0 # 左端で停止
            
        else:
            next_state = state-1 
        reward = -0.5 # 移動ペナルティ
    return next_state, reward

def update_env(state,episode, step_counter):
    """
    環境の表示更新(進行状況の可視化)
    """
    env = ['-'] *(N_STATE-1)+['T'] # Tはゴール地点
    if state =='terminal':
        print("Episode {}, the total step is {}".format(episode+1, step_counter))
        final_env = ['-'] *(N_STATE-1)+['T']
        return True, step_counter # エピソード終了
    else:
        env[state]='*' # エージェントの位置を表示
        env = ''.join(env)
        print(env)
        time.sleep(FRESH_TIME)
        return False, step_counter
        
    
def q_learning():
    """
    Q学習のメイン処理
    """
    q_table = build_q_table(N_STATE, ACTIONS) # Qテーブル初期化
    step_counter_times = [] # 各エピソードのステップ数を記録

    for episode in range(MAX_EPISODES):
        state = 0 # エピソード開始時の初期状態
        is_terminal = False
        step_counter = 0
        update_env(state, episode, step_counter)
        while not is_terminal:
            action = choose_action(state,q_table) # 行動選択
            next_state, reward = get_env_feedback(state, action) # 環境応答取得
            next_q = q_table.loc[state, action] # 現在の状態と行動からQ値を取得

            # Q値更新
            if next_state == 'terminal': # ゴールに到着するとき
                is_terminal = True
                q_target = reward
            else:
                # TD誤差と更新
                delta = reward + GAMMA*q_table.iloc[next_state,:].max()-q_table.loc[state, action]
                q_table.loc[state, action] += ALPHA*delta

            state = next_state # 状態遷移
            is_terminal,steps = update_env(state, episode, step_counter+1)
            step_counter+=1
            if is_terminal:
                step_counter_times.append(steps) # ステップ数記録
                
    return q_table, step_counter_times

def main():
    q_table, step_counter_times= q_learning()
    print("Q table\n{}\n".format(q_table)) # 最終Qテーブル出力
    print('end')
    
    plt.plot(step_counter_times,'g-')
    plt.ylabel("steps")
    plt.show()
    print("The step_counter_times is {}".format(step_counter_times))

main() 