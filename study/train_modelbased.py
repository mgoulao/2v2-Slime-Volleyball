"""
Train PPO using Selfplay

run: python train_ppo.py

Train a PPO policy using Selfplay

"""
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import argparse

from agents_wraps.modelbased import Model_Team
from selfplay import SlimeVolleySelfPlayEnv

RENDER_MODE = False
SELFPLAY = True

LOGDIR = "./model_saves"

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train PPO.')
    parser.add_argument('--render', action='store_true', help='Enable environment render', default=False)
    parser.add_argument('--noselfplay', action='store_true', help='Disable selfplay', default=False)
    args = parser.parse_args()

    RENDER_MODE = args.render
    SELFPLAY = not args.noselfplay

    env = SlimeVolleySelfPlayEnv(Model_Team, LOGDIR, RENDER_MODE, SELFPLAY)
    teamPPO = Model_Team(env, LOGDIR)
    teamPPO.loadBestModel()

    teamPPO.train(int(1e7))
    teamPPO.save("selfplay_model")

