
import numpy as np

class GridWorld:

    def __init__(self, tot_row, tot_col):
        # 行動空間のサイズ（上，右，下，左の4つ）
        self.action_space_size = 4
        self.world_row = tot_row
        self.world_col = tot_col
        
        # 遷移確率行列（各行動ごとに実際に取られる行動の確率）
        # 初期状態では全行動均等確率で行動が取られる
        self.transition_matrix = np.ones((self.action_space_size, self.action_space_size))/ self.action_space_size
        
        # 各マスの報酬
        self.reward_matrix = np.zeros((tot_row, tot_col))

        # 各マスの状態種別
        self.state_matrix = np.zeros((tot_row, tot_col))

        # エージェントの初期位置をランダムに設定
        self.position = [np.random.randint(tot_row), np.random.randint(tot_col)]   

    def setTransitionMatrix(self, transition_matrix):
        """
        transition_matrix は 4x4 の行列で、各行は指定行動に対して実際にどの行動が取られるかの確率を表す。
        例:
        [[0.55, 0.25, 0.10, 0.10],  # UPに対して
         [0.25, 0.25, 0.25, 0.25],  # RIGHTに対して
         ... ]
        """
        if(transition_matrix.shape != self.transition_matrix.shape):
            raise ValueError('The shape of the two matrices must be the same.') 
        self.transition_matrix = transition_matrix

    def setRewardMatrix(self, reward_matrix):
        """
        報酬行列の設定
        各マスに設定する報酬値を渡す
        """
        if(reward_matrix.shape != self.reward_matrix.shape):
            raise ValueError('The shape of the matrix does not match with the shape of the world.')
        self.reward_matrix = reward_matrix

    def setStateMatrix(self, state_matrix):
        """
        各マスの状態を設定する行列
        -1: 通行不可（障害物）
         0: 通常状態（通行可能）
        +1: 終端状態（ゴール）
        """
        if(state_matrix.shape != self.state_matrix.shape):
            raise ValueError('The shape of the matrix does not match with the shape of the world.')
        self.state_matrix = state_matrix

    def setPosition(self, index_row=None, index_col=None):
        """
        エージェントの位置を設定
        """
        if(index_row is None or index_col is None): self.position = [np.random.randint(tot_row), np.random.randint(tot_col)]
        else: self.position = [index_row, index_col]

    def render(self):
        """
        グリッドワールドをターミナル上に表示：
        ○：エージェントの現在位置
        -：通常マス
        #：障害物
        *：終端状態
        """
        graph = ""
        for row in range(self.world_row):
            row_string = ""
            for col in range(self.world_col):
                if(self.position == [row, col]): row_string += u" \u25CB " # u" \u25CC "
                else:
                    if(self.state_matrix[row, col] == 0): row_string += ' - '
                    elif(self.state_matrix[row, col] == -1): row_string += ' # '
                    elif(self.state_matrix[row, col] == +1): row_string += ' * '
            row_string += '\n'
            graph += row_string 
        print(graph)            

    def reset(self, exploring_starts=False):
        """
        環境をリセットして、エージェントの位置を初期化する。
        exploring_starts = True の場合は、ランダムな通常マスに配置。
        """
        if exploring_starts:
            while(True):
                row = np.random.randint(0, self.world_row)
                col = np.random.randint(0, self.world_col)
                if(self.state_matrix[row, col] == 0): break # 通常ますならOK
            self.position = [row, col]
        else:
            self.position = [self.world_row-1, 0] # 左下固定スタート
        return self.position

    def step(self, action):
        """
        与えられた行動に基づいて次の状態に遷移し、報酬とdoneフラグを返す。
        行動: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

        戻り値：
        - observation: 次の位置 [row, col]
        - reward: 次状態の報酬
        - done: ゴールまたは終端状態に到達したかどうか
        """
        if(action >= self.action_space_size): 
            raise ValueError('The action is not included in the action space.')

        # 遷移確率に従って実際にとる行動をサンプリング
        action = np.random.choice(4, 1, p=self.transition_matrix[int(action),:])

        # 新しい位置を計算
        if(action == 0): new_position = [self.position[0]-1, self.position[1]]   #UP
        elif(action == 1): new_position = [self.position[0], self.position[1]+1] #RIGHT
        elif(action == 2): new_position = [self.position[0]+1, self.position[1]] #DOWN
        elif(action == 3): new_position = [self.position[0], self.position[1]-1] #LEFT
        else: raise ValueError('The action is not included in the action space.')

        # 境界外または障害物でなければ位置を更新
        if (new_position[0]>=0 and new_position[0]<self.world_row):
            if(new_position[1]>=0 and new_position[1]<self.world_col):
                if(self.state_matrix[new_position[0], new_position[1]] != -1):
                    self.position = new_position

        # 報酬と終了判定を取得
        reward = self.reward_matrix[self.position[0], self.position[1]]
        done = bool(self.state_matrix[self.position[0], self.position[1]])
        return self.position, reward, done