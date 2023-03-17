import argparse
import gym
from gym.core import ActType, ObsType
from gym.spaces import Discrete, Box
import numpy as np
import os
import random

from ray.rllib.env import EnvContext
import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.ppo import PPOConfig
import warnings

from gym_connect_four.envs.connect_four_env import ConnectFourEnv

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

# import custom environment
from model import CustomModel
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm

ray.init()

config = PPOConfig()

config = config.rollouts(num_rollout_workers=1)

config = config.environment(ConnectFourEnv, env_config={"human_play": True, "greedy_train": False})

# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build()
path = "checkpoints/"
subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
# print(subfolders)
max_time = 0
newest_checkpoint = ""
for subfolder in subfolders:
    # print(subfolder)
    st_mtime = os.stat(subfolder).st_mtime
    if st_mtime > max_time:
        max_time = st_mtime
        newest_checkpoint = subfolder
print("newest checkpoint:", str(newest_checkpoint))
algo.restore(newest_checkpoint)

train = algo.train()
print(train)

