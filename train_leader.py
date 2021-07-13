"""
Train PPO with Leader/Teammate roles using Selfplay

run: python train_leader.py
"""
import argparse

from agents.ppo_leader import LeaderTeam
from selfplay import SlimeVolleySelfPlayEnv

RENDER_MODE = False
SELFPLAY = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO.')
    parser.add_argument('--render', action='store_true', help='Enable environment render', default=False)
    parser.add_argument('--noselfplay', action='store_true', help='Disable selfplay', default=False)
    args = parser.parse_args()

    RENDER_MODE = args.render
    SELFPLAY = not args.noselfplay

    env = SlimeVolleySelfPlayEnv(LeaderTeam, RENDER_MODE, SELFPLAY)
    teamPPO = LeaderTeam(env)
    teamPPO.loadBestModel()

    teamPPO.train(int(1e7))
    teamPPO.save("selfplay_leader")

