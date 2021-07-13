"""
Evaluate agents in a 2v2 slimevolley game
"""

import numpy as np
import argparse
import slimevolleygym 
from slimevolleygym import BaselinePolicy

from agents.ppo import PPOTeam
from agents.ppo_ad import ADTeam
from agents.ppo_top_bot import TopBotTeam
from agents.ppo_leader import LeaderTeam

class MultiAgentBaselinePolicy(BaselinePolicy):
    def __init__(self, env):
        super(MultiAgentBaselinePolicy, self).__init__()
        self.agent1 = None

    def predict(self, obs_1, obs_2):
        return super().predict(obs_1), super().predict(obs_2)
class SlimeVolleyEvalEnv(slimevolleygym.SlimeVolleyEnv):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, renderMode):
        super(SlimeVolleyEvalEnv, self).__init__()
        self.policy = self
        self.opponent = None
        self.renderMode = renderMode
        self.survival_bonus = False

    def set_opponent(self, opponent):
        self.opponent = opponent

    def predict(self, obs1, obs2): # Opponent policy
        if self.opponent == None:
            print("Please set the opponent first")
            return

        action1, action2 = self.opponent.predict(obs1, obs2)
        return action1, action2

    def step(self, action):
        if self.renderMode:
            self.render()
        return super(SlimeVolleyEvalEnv, self).step(action)


def rollout(env, policy_1):
  """ play one agent vs the other in modified gym-style loop. """
  obs_1, obs_2 = env.reset()

  done = False
  total_reward = 0

  while not done:

    action_1, action_2 = policy_1.predict(obs_1, obs_2)
    obs_arr, reward, done, _ = env.step([action_1, action_2])
    obs_1 = obs_arr[0]
    obs_2 = obs_arr[1]

    total_reward += reward

  return total_reward

def evaluate_multiagent(env, policy_1, n_trials=500, init_seed=721):
    print("2v2 Slimevolley Evaluation")
    history = []
    for i in range(1, n_trials+1):
        env.seed(seed=init_seed+i)
        cumulative_score = rollout(env, policy_1)
        print("cumulative score #", i, ":", cumulative_score)
        history.append(cumulative_score)


    return history

if __name__=="__main__":

    APPROVED_MODELS = ["baseline", "ppo", "ppo_ad", "ppo_top_bot", "ppo_leader"]

    def check_choice(choice):
        return choice.lower() in APPROVED_MODELS

    MODEL = {
        "baseline": MultiAgentBaselinePolicy,
        "ppo": PPOTeam,
        "ppo_ad": ADTeam,
        "ppo_top_bot": TopBotTeam,
        "ppo_leader": LeaderTeam
    }

    parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
    parser.add_argument('--left', help='choice of (baseline, ppo, ppo_ad, ppo_top_bot, ppo_leader)', type=str, default="baseline")
    parser.add_argument('--right', help='choice of (baseline, ppo, ppo_ad, ppo_top_bot, ppo_leader)', type=str, default="ppo")
    parser.add_argument('--render', action='store_true', help='render to screen?', default=False)

    args = parser.parse_args()
    render_mode = args.render
    
    right_agent_choice = args.right
    left_agent_choice = args.left

    assert check_choice(args.right), "Please enter a valid agent"
    assert check_choice(args.left), "Please enter a valid agent"
    
    env = SlimeVolleyEvalEnv(render_mode)

    policy_1 = MODEL[right_agent_choice](env) # the right agent
    policy_2 = MODEL[left_agent_choice](env) # the left agent

    if not right_agent_choice == "baseline":
        policy_1.loadBestModel()

    if not left_agent_choice == "baseline":
        policy_2.loadBestModel()

    env.set_opponent(policy_2)
    history = evaluate_multiagent(env, policy_1)

    print("history dump:", history)
    print(right_agent_choice + " scored", np.round(np.mean(history), 3), "+/-", np.round(np.std(history), 3), "vs",
        left_agent_choice, "over")
