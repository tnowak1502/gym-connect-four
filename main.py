import gym
from gym_connect_four import RandomPlayer, ConnectFourEnv
from greedy import GreedyPlayer
env: ConnectFourEnv = gym.make("ConnectFour-v0")

player1 = GreedyPlayer(env, 'Dexter-Bot')
player2 = GreedyPlayer(env, 'Deedee-Bot')
result = env.run(player1, player2, render=True)
reward = result.value
print(reward)