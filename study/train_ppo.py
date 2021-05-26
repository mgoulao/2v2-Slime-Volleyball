"""
Train PPO using Selfplay

run: python train_ppo.py

Train a PPO policy using Selfplay

"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import argparse

from agents_wraps.ppo2 import PPO_TEAM
from selfplay import SlimeVolleySelfPlayEnv

RENDER_MODE = False
SELFPLAY = True

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train PPO.')
    parser.add_argument('--render', action='store_true', help='Enable environment render', default=True)
    parser.add_argument('--noselfplay', action='store_true', help='Disable selfplay', default=False)
    args = parser.parse_args()

    RENDER_MODE = args.render
    SELFPLAY = not args.noselfplay

    env = SlimeVolleySelfPlayEnv(PPO_TEAM, RENDER_MODE, SELFPLAY)
    teamPPO = PPO_TEAM(env)
    teamPPO.loadBestModel()

    teamPPO.train(int(1e7))
    teamPPO.save("selfplay_ppo")

