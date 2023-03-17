import random
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, unique
from operator import itemgetter
from typing import Tuple, NamedTuple, Hashable, Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import error
from gymnasium import spaces
from tensorflow.keras.models import load_model

from gym_connect_four.envs.render import render_board
#from greedy import GreedyPlayer

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
import os
from ray.rllib.env.env_context import EnvContext
from PIL import Image

class Player(ABC):
    """ Class used for evaluating the game """

    def __init__(self, env: 'ConnectFourEnv', name='Player'):
        self.name = name
        self.env = env

    @abstractmethod
    def get_next_action(self, state: np.ndarray) -> int:
        pass

    def learn(self, state, action: int, state_next, reward: int, done: bool) -> None:
        pass

    def save_model(self, model_prefix: str = None):
        raise NotImplementedError()

    def load_model(self, model_prefix: str = None):
        raise NotImplementedError()

    def reset(self, episode: int = 0, side: int = 1) -> None:
        """
        Allows a player class to reset it's state before each round

            Parameters
            ----------
            episode : which episode we have reached
            side : 1 if the player is starting or -1 if the player is second
        """
        pass


class RandomPlayer(Player):
    def __init__(self, env: 'ConnectFourEnv', name='RandomPlayer', seed: Optional[Hashable] = None):
        super().__init__(env, name)
        self._seed = seed
        # For reproducibility of the random
        prev_state = random.getstate()
        random.seed(self._seed)
        self._state = random.getstate()
        random.setstate(prev_state)

    def get_next_action(self, state: np.ndarray) -> int:
        available_moves = self.env.available_moves()
        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')

        # Next operations are needed for reproducibility of the RandomPlayer when inited with seed
        prev_state = random.getstate()
        random.setstate(self._state)
        action = random.choice(list(available_moves))
        self._state = random.getstate()
        random.setstate(prev_state)
        return action

    def reset(self, episode: int = 0, side: int = 1) -> None:
        # For reproducibility of the random
        random.seed(self._seed)
        self._state = random.getstate()

    def save_model(self, model_prefix: str = None):
        pass


class SavedPlayer(Player):
    def __init__(self, env, name='SavedPlayer', model_prefix=None):
        super(SavedPlayer, self).__init__(env, name)

        if model_prefix is None:
            model_prefix = self.name

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.model = load_model(f"{model_prefix}.h5")

    def get_next_action(self, state: np.ndarray) -> int:
        state = np.reshape(state, [1] + list(self.observation_space))
        q_values = self.model.predict(state)[0]
        vs = [(i, q_values[i]) for i in self.env.available_moves()]
        act = max(vs, key=itemgetter(1))
        return act[0]


@unique
class ResultType(Enum):
    NONE = None
    DRAW = 0
    WIN1 = 1
    WIN2 = -1
    INVALID = -2

    def __eq__(self, other):
        """
        Need to implement this due to an unfixed bug in Python since 2017: https://bugs.python.org/issue30545
        """
        return self.value == other.value


class ConnectFourEnv(gym.Env):
    """
    Description:
        ConnectFour game environment

    Observation:
        Type: Discreet(6,7)

    Actions:
        Type: Discreet(7)
        Num     Action
        x       Column in which to insert next token (0-6)

    Reward:
        Reward is 0 for every step.
        If there are no other further steps possible, Reward is 0.5 and termination will occur
        If it's a win condition, Reward will be 1 and termination will occur
        If it is an invalid move, Reward will be -1 and termination will occur

    Starting State:
        All observations are assigned a value of 0

    Episode Termination:
        No more spaces left for pieces
        4 pieces are present in a line: horizontal, vertical or diagonally
        An attempt is made to place a piece in an invalid location
    """

    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    LOSS_REWARD = -1
    DEF_REWARD = 0
    DRAW_REWARD = 0.5
    WIN_REWARD = 1
    INVALID_REWARD = 0

    class StepResult(NamedTuple):

        res_type: ResultType

        def get_reward(self, player: int):
            if self.res_type is ResultType.INVALID:
                return ConnectFourEnv.INVALID_REWARD
            elif self.res_type is ResultType.NONE:
                return ConnectFourEnv.DEF_REWARD
            elif self.res_type is ResultType.DRAW:
                return ConnectFourEnv.DRAW_REWARD
            else:
                return {ResultType.WIN1.value: ConnectFourEnv.WIN_REWARD, ResultType.WIN2.value: ConnectFourEnv.LOSS_REWARD}[
                    self.res_type.value * player]

        def is_done(self):
            return self.res_type != ResultType.NONE and self.res_type != ResultType.INVALID

    def __init__(self, env_config: EnvContext, board_shape=(6, 7), window_width=512, window_height=512):
        #print("WINDOW WIDTH", window_width)
        #print("BOARD SHAPE", str(board_shape))
        super(ConnectFourEnv, self).__init__()

        self.board_shape = (6, 7)
        #print("BOARD SHAPE", self.board_shape)

        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=self.board_shape,
                                            dtype=int)
        self.action_space = spaces.Discrete(self.board_shape[1])

        self.__current_player = 1
        self.__board = np.zeros(self.board_shape, dtype=int)

        self.__player_color = 1
        self.__screen = None
        self.__window_width = window_width
        self.__window_height = window_height
        self.__rendered_board = self._update_board_render()
        self.human_play = env_config["human_play"]
        self.step_count = 0
        self.greedy_train = env_config["greedy_train"]
        if self.human_play and self.greedy_train:
            print("invalid config, setting human play to false")
            self.human_play = false
        print("Human Play", self.human_play)
        print("greedy_train", self.greedy_train)
        print("loading algo in connect four")
        # Build the Algorithm instance using the config.
        # Restore the algo's state from the checkpoint.
        if not self.greedy_train:
            path = "checkpoints/"
            subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
            # print(subfolders)
            max_time = 0
            newest_checkpoint = ""
            for subfolder in subfolders:
                #print(subfolder)
                st_mtime = os.stat(subfolder).st_mtime
                if st_mtime > max_time:
                    max_time = st_mtime
                    newest_checkpoint = subfolder
            print("ENV: newest checkpoint:", str(newest_checkpoint))
            self.policy = Policy.from_checkpoint(str(newest_checkpoint) +
                "/policies/default_policy")
            print("algo loaded")
        else:
            self.policy = ""
    # def copy(self, board):
    #     newEnv = gym.make("ConnectFour-v0")
    #     newEnv.board = board
    #     return newEnv

    def run(self, player1: Player, player2: Player, board: Optional[np.ndarray] = None, render=False) -> ResultType:
        player1.reset()
        player2.reset()
        self.reset(board)

        cp = lambda: self.__current_player

        def change_player():
            self.__current_player *= -1
            return player1 if cp() == 1 else player2

        state_hist = deque([self.__board.copy()], maxlen=4)

        act = player1.get_next_action(self.__board * 1)
        act_hist = deque([act], maxlen=2)
        step_result = self._step(act)
        state_hist.append(self.__board.copy())
        player = change_player()
        done = False
        while not done:
            if render:
                self.render()
            act_hist.append(player.get_next_action(self.__board * cp()))
            step_result = self._step(act_hist[-1])
            state_hist.append(self.__board.copy())

            player = change_player()

            reward = step_result.get_reward(cp())
            done = step_result.is_done()
            player.learn(state=state_hist[-3] * cp(), action=act_hist[-2], state_next=state_hist[-1] * cp(), reward=reward, done=done)

        player = change_player()
        reward = step_result.get_reward(cp())
        player.learn(state_hist[-2] * cp(), act_hist[-1], state_hist[-1] * cp(), reward, done)
        if render:
            self.render()

        return step_result.res_type

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        #print("Agent step", self.__current_player)
        board, step_result = self._step(action, self.__board.copy())
        self.__board = board
        #print(step_result.is_done())
        if step_result.res_type is ResultType.INVALID:
            #print("INVALID MOVE!!!!!!!!!!!!!!!!!!!!!")
            reward = step_result.get_reward(self.__current_player)
            return self.__board.copy(), reward, False, False, {}
        if step_result.is_done():
            #print(self.is_win_state(self.__board), step_result.is_done(), self.__current_player)
            reward = step_result.get_reward(self.__current_player)
            self.__board = board
            #print(self.__board)
            #self.render()
            #self._update_board_render()
            return self.__board.copy(), reward, True, False, {}
        self.__current_player *= -1

        if self.greedy_train:
            #print("Greedy Step", self.__current_player)
            bestMove = 0
            bestValue = 0
            for action in range(self.board_shape[1]):
                if not self.is_valid_action(action):
                    continue
                test_board = self.__board.copy()
                test_board, step_result = self._step(action, self.__board.copy())
                longestChain = 0
                if step_result.is_done():
                    bestMove = action
                    break
                for i in range(test_board.shape[0]):
                    current_chain = 0
                    for j in range(test_board.shape[1]):
                        if test_board[i][j] == 1:
                            current_chain += 1
                        elif test_board[i][j] == 0:
                            if current_chain > longestChain:
                                longestChain = current_chain
                            current_chain = 0
                        else:
                            current_chain = 0
                # check columns
                trans_board = np.transpose(test_board)
                for i in range(test_board.shape[1]):
                    current_chain = 0
                    for j in range(test_board.shape[0]):
                        if trans_board[i][j] == 1:
                            current_chain += 1
                        elif trans_board[i][j] == 0:
                            if current_chain > longestChain:
                                longestChain = current_chain
                            current_chain = 0
                        else:
                            current_chain = 0
                # check diagonals
                for i in range(test_board.shape[0]):
                    current_chain = 0
                    for j in range(test_board.shape[1]):
                        while i < test_board.shape[0] and j < test_board.shape[1]:
                            if test_board[i][j] == 1:
                                current_chain += 1
                            elif test_board[i][j] == 0:
                                if current_chain > longestChain:
                                    longestChain = current_chain
                                current_chain = 0
                            else:
                                current_chain = 0
                            i += 1
                            j += 1
                # check reversed diagonals
                flipped_board = np.fliplr(test_board)
                for i in range(test_board.shape[0]):
                    current_chain = 0
                    for j in range(test_board.shape[1]):
                        while i < test_board.shape[0] and j < test_board.shape[1]:
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
        elif not self.human_play:
            bestMove = self.policy.compute_single_action(self.__board.copy())[0]
        #print("bestMove", bestMove)
        else:
            self.render(mode="console")
            im = Image.fromarray(self.render(mode="rgb_array"))
            im.save("episode_images/" + str(self.step_count) + ".jpg")
            self.step_count += 1
            print("Make your move (0-6)")
            bestMove = input()
            while(not bestMove.isdigit() or int(bestMove) < 0 or int(bestMove) > self.board_shape[0]):
                print("Please enter a valid move (0-6)")
                bestMove = input()
            bestMove = int(bestMove)

        board, step_result = self._step(bestMove, self.__board.copy())
        self.__board = board
        #print(self.is_win_state(self.__board), step_result.is_done(), self.__current_player)
        self.__current_player *= -1
        reward = step_result.get_reward(self.__current_player)
        done = step_result.is_done()
        if self.human_play:
            self.render(mode="console")
            im = Image.fromarray(self.render(mode="rgb_array"))
            im.save("episode_images/" + str(self.step_count) + ".jpg")
            self.step_count += 1
        #print(self.__board)
        return self.__board.copy(), reward, done, False, {}

    def _step(self, action, board):
        result = ResultType.NONE

        if not self.is_valid_action(action):
            result = ResultType.INVALID
            return board, self.StepResult(result)

        # Check and perform action
        for index in list(reversed(range(self.board_shape[0]))):
            if board[index][action] == 0:
                board[index][action] = self.__current_player
                break

        # Check if board is completely filled
        if np.count_nonzero(board[0]) == self.board_shape[1]:
            result = ResultType.DRAW
        else:
            # Check win condition
            if self.is_win_state(board):
                result = ResultType.WIN1 if self.__current_player == 1 else ResultType.WIN2
        return board, self.StepResult(result)

    @property
    def board(self):
        return self.__board.copy()

    def reset(self, board: Optional[np.ndarray] = None, seed=None, options=None) -> (np.ndarray, dict):
        self.step_count = 0
        self.__current_player = 1
        if board is None:
            self.__board = np.zeros(self.board_shape, dtype=int)
        else:
            self.__board = board
        self.__rendered_board = self._update_board_render()
        return self.board, {}

    def render(self, mode: str = 'console', close: bool = False) -> None:
        if mode == 'console':
            replacements = {
                self.__player_color: 'A',
                0: ' ',
                -1 * self.__player_color: 'B'
            }

            def render_line(line):
                return "|" + "|".join(
                    ["{:>2} ".format(replacements[x]) for x in line]) + "|"

            hline = '|---+---+---+---+---+---+---|'
            print(hline)
            for line in np.apply_along_axis(render_line,
                                            axis=1,
                                            arr=self.__board):
                print(line)
            print(hline)

        elif mode == 'human':
            if self.__screen is None:
                pygame.init()
                self.__screen = pygame.display.set_mode(
                    (round(self.__window_width), round(self.__window_height)))

            if close:
                pygame.quit()

            self.__rendered_board = self._update_board_render()
            frame = self.__rendered_board
            surface = pygame.surfarray.make_surface(frame)
            surface = pygame.transform.rotate(surface, 90)
            self.__screen.blit(surface, (0, 0))

            pygame.display.flip()
        elif mode == 'rgb_array':
            self.__rendered_board = self._update_board_render()
            frame = self.__rendered_board
            return np.fliplr(np.flip(frame, axis=(0, 1)))
        else:
            raise error.UnsupportedMode()

    def close(self) -> None:
        pygame.quit()
        self.__screen = None

    def is_valid_action(self, action: int) -> bool:
        return self.__board[0][action] == 0

    def _update_board_render(self) -> np.ndarray:
        return render_board(self.__board,
                            image_width=self.__window_width,
                            image_height=self.__window_height)

    def is_win_state(self, board) -> bool:
        # Test rows
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1] - 3):
                value = sum(board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test columns on transpose array
        reversed_board = [list(i) for i in zip(*board)]
        for i in range(self.board_shape[1]):
            for j in range(self.board_shape[0] - 3):
                value = sum(reversed_board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        reversed_board = np.fliplr(board)
        # Test reverse diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += reversed_board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        return False

    def available_moves(self):
        return (i for i in range(self.board_shape[1]) if self.is_valid_action(i))