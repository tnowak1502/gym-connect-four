import matplotlib
matplotlib.use('Agg')
import numpy as np
from gym_connect_four.envs.connect_four_env import ConnectFourEnv, Player, ResultType, SavedPlayer, RandomPlayer

class GreedyPlayer(Player):
    def __init__(self, env, name='GreedyPlayer'):
        super(GreedyPlayer, self).__init__(env, name)
        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n
    def get_next_action(self, state: np.ndarray) -> int:
        state = np.reshape(state, [1] + list(self.observation_space))
        bestMove = 0
        bestValue = 0
        for action in range(self.action_space):
            if not self.env.is_valid_action(action):
                continue
            test_env = self.env.copy(self.env.board)
            longestChain = 0
            test_env.reset()
            newState = test_env.step(action)
            if newState[1] == 1:
                bestMove = action
                break
            elif newState[1] == -1:
                continue
            else:
                board = newState[0]
                #check rows
                for i in range(self.observation_space[0]):
                    current_chain = 0
                    for j in range(self.observation_space[1]):
                        if board[i][j] == 1:
                            current_chain += 1
                        elif board[i][j] == 0:
                            if current_chain > longestChain:
                                longestChain = current_chain
                            current_chain = 0
                        else:
                            current_chain = 0
                #check columns
                trans_board = np.transpose(board)
                for i in range(self.observation_space[1]):
                    current_chain = 0
                    for j in range(self.observation_space[0]):
                        if trans_board[i][j] == 1:
                            current_chain += 1
                        elif trans_board[i][j] == 0:
                            if current_chain > longestChain:
                                longestChain = current_chain
                            current_chain = 0
                        else:
                            current_chain = 0
                #check diagonals
                for i in range(self.observation_space[0]):
                    current_chain = 0
                    for j in range(self.observation_space[1]):
                        while i < self.observation_space[0] and j < self.observation_space[1]:
                            if board[i][j] == 1:
                                current_chain += 1
                            elif board[i][j] == 0:
                                if current_chain > longestChain:
                                    longestChain = current_chain
                                current_chain = 0
                            else:
                                current_chain = 0
                            i += 1
                            j += 1
                #check reversed diagonals
                flipped_board = np.fliplr(board)
                for i in range(self.observation_space[0]):
                    current_chain = 0
                    for j in range(self.observation_space[1]):
                        while i < self.observation_space[0] and j < self.observation_space[1]:
                            if flipped_board[i][j] == 1:
                                current_chain += 1
                            elif flipped_board[i][j] == 0:
                                if current_chain > longestChain:
                                    longestChain = current_chain
                                current_chain = 0
                            else:
                                current_chain = 0
                            i += 1
                            j += 1
                if longestChain > bestValue:
                    bestMove = action
                    bestValue = longestChain

        action = bestMove
        if self.env.is_valid_action(action):
            return action

