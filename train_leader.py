"""
Train PPO using Selfplay

run: python train_ppo.py

Train a PPO policy using Selfplay

"""
import argparse

from agents.ppo_leader import LEADER_TEAM
from selfplay import SlimeVolleySelfPlayEnv

RENDER_MODE = False
SELFPLAY = True

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train PPO.')
    parser.add_argument('--render', action='store_true', help='Enable environment render', default=False)
    parser.add_argument('--noselfplay', action='store_true', help='Disable selfplay', default=False)
    args = parser.parse_args()

    RENDER_MODE = args.render
    SELFPLAY = not args.noselfplay

    env = SlimeVolleySelfPlayEnv(LEADER_TEAM, RENDER_MODE, SELFPLAY)
    teamPPO = LEADER_TEAM(env)
    teamPPO.loadBestModel()

    teamPPO.train(int(1e7))
    teamPPO.save("selfplay_leader")

