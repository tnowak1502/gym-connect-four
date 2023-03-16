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

ray.init()
config = PPOConfig()

config = config.rollouts(num_rollout_workers=1)

# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build(env=ConnectFourEnv)

for i in range(2):
    print(algo.train())

checkpoint_dir = algo.save("C:/Users/thoma/Documents/Uni/Teamprojekt/rllib_checkpoint")
# class Arguments():
#     # default values, previously set via cmd line parameters
#     #
#     # to be implemented: store configuration in a dedicated file (e.g., JSON) and import settings here
#     def __init__(self) -> None:
#         # sets the optimization algorithm, in this case proximal policy optimization
#         # can be adapted to run customized learning algorithms as well
#         self.run = "PPO"
#         # hyperparameters for customized training
#         self.framework = "tf"
#         self.as_test = True
#         self.stop_iters = 50
#         self.stop_timesteps = 100000
#         self.stop_reward = 1
#         # set to false in order to enable result logging
#         self.no_tune = False
#         self.local_mode = True
#         # custom option added to utilize GPUs (if available, not working yet)
#         self.use_gpus = 0
#         # set variable to turn visualization on/off
#         self.render = False
#
# args = Arguments()
#
# # seems not to be working, figure out another solution
# warnings.filterwarnings("ignore", category=DeprecationWarning)
#
# if __name__ == "__main__":
#     print(f"Running with following CLI options: {args}")
#
#     ray.init(local_mode=args.local_mode, num_gpus=1)
#
#     # Can also register the env creator function for custom encironemnts explicitly with:
#     # register_env("corridor", lambda config: AvalancheEnv(config))
#     ModelCatalog.register_custom_model(
#         "my_model", CustomModel
#     )
#
#     config = (
#         get_trainable_cls(args.run)
#             .get_default_config()
#             # or "corridor" if registered above
#             # set render_env to true, if a live visualization of the training process is required
#             .environment(ConnectFourEnv)#, render_env=True)
#             .framework(args.framework)
#             .rollouts(num_rollout_workers=1)
#             .training(
#                 model={
#                     "custom_model": "my_model",
#                     "vf_share_layers": True,
#                 }
#             )
#             .evaluation(
#             # Evaluate once per training iteration.
#             evaluation_interval=1,
#             # Run evaluation on (at least) two episodes
#             evaluation_duration=2,
#             # ... using one evaluation worker (setting this to 0 will cause
#             # evaluation to run on the local evaluation worker, blocking
#             # training until evaluation is done).
#             evaluation_num_workers=1,
#             # Special evaluation config. Keys specified here will override
#             # the same keys in the main config, but only for evaluation.
#             evaluation_config={
#                 # Render the env while evaluating.
#                 # Note that this will always only render the 1st RolloutWorker's
#                 # env and only the 1st sub-env in a vectorized env.
#                 "render_env": args.render
#             },
#         )
#             # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
#             .resources(num_gpus=args.use_gpus)#int(os.environ.get("RLLIB_NUM_GPUS", "0")))
#     )
#
#     stop = {
#         "training_iteration": args.stop_iters,
#         "timesteps_total": args.stop_timesteps,
#         "episode_reward_mean": args.stop_reward,
#     }
#
#     # automated run with Tune and grid search and TensorBoard
#     print("Training automatically with Ray Tune")
#     tuner = tune.Tuner(
#         args.run,
#         param_space=config.to_dict(),
#         run_config=air.RunConfig(stop=stop, local_dir="./results", name="test_experiment"),
#     )
#     results = tuner.fit()
#
#     if args.as_test:
#         print("Checking if learning goals were achieved")
#         check_learning_achieved(results, args.stop_reward)
#
#     ray.shutdown()
#
#     # high-level result interpretation (if no_tune == False, stored in results folder)
#     # training is srtuctured in episodes
#     # episode = one loop of training/rewards/optimization until defined end state
#     # in the example of CartPole, one end state is, e.g., if the cart position is outside the display zone
#
#
#     # for more details, visit https://docs.ray.io/en/latest/rllib/core-concepts.html