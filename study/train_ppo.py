"""
Train PPO using Selfplay

run: python train_ppo.py --render

Train a PPO policy using Selfplay

"""
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import argparse

from agents_wraps.ppo2 import PPO_TEAM
from selfplay import SlimeVolleySelfPlayEnv

RENDER_MODE = False
SELFPLAY = True

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
        #self.agent2 = PPO("MlpPolicy", env, verbose=1)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train PPO.')
    parser.add_argument('--render', action='store_true', help='Enable environment render', default=False)
    parser.add_argument('--noselfplay', action='store_true', help='Disable selfplay', default=False)
    args = parser.parse_args()

    RENDER_MODE = args.render
    SELFPLAY = not args.noselfplay

    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    env = SlimeVolleySelfPlayEnv(LOGDIR, RENDER_MODE, SELFPLAY)
    teamPPO = PPO_TEAM(env, LOGDIR)
    teamPPO.loadBestModel()

    teamPPO.train(int(1e7))
    

