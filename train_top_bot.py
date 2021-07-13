"""
Train PPO with Top/Bot roles using Selfplay

run: python train_top_bot.py
"""
import argparse

from agents.ppo_top_bot import TopBotTeam
from selfplay import SlimeVolleySelfPlayEnv

RENDER_MODE = False
SELFPLAY = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO with roles.')
    parser.add_argument('--render', action='store_true', help='Enable environment render', default=True)
    parser.add_argument('--noselfplay', action='store_true', help='Disable selfplay', default=False)
    args = parser.parse_args()

    RENDER_MODE = args.render
    SELFPLAY = not args.noselfplay

    env = SlimeVolleySelfPlayEnv(TopBotTeam, RENDER_MODE, SELFPLAY)
    teamPPO = TopBotTeam(env)
    teamPPO.loadBestModel()

    teamPPO.train(int(1e7))
    teamPPO.save("selfplay_roles")
