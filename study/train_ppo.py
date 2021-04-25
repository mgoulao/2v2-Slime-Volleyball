"""
Simple evaluation example.

run: python eval_ppo.py --render

Evaluate PPO1 policy (MLP input_dim x 64 x 64 x output_dim policy) against built-in AI

"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import argparse

from agents_wraps.ppo import PPO_TEAM
from selfplay import SelfPlayCallback, SlimeVolleySelfPlayEnv

SEED = 17
NUM_TIMESTEPS = int(1e9)
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(1e2)

RENDER_MODE = False

LOGDIR = "./ppo_logs"

class MODEL_TEAM:
    def __init__(self, env):
        pass
        self.agent1 = None #SARSA
        self.agent2 = None #SARSA

class OUR_TEAM:
    def __init__(self, env):
        pass
        #self.agent1 = PPO("MlpPolicy", env, verbose=1)
        #self.agent2 = PPO("MlpPolicy", env, verbose=1) # TODO: change state to add agent1's action

if __name__=="__main__":
    env = SlimeVolleySelfPlayEnv(LOGDIR)
    teamPPO = PPO_TEAM("MlpPolicy", env)
    eval_callback = SelfPlayCallback(env,
        best_model_save_path=LOGDIR,
        log_path=LOGDIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=False)

    teamPPO.learn(10000, callback=eval_callback)

    

