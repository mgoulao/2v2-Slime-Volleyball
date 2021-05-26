"""
Test different agents in a 2v2 slimevolley game

Model Choices
=============
baseline: Baseline Policy (built-in AI). Simple 120-param RNN.
ppo: PPO trained using selfplay
"""

import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import sys
sys.path.append('../slimevolleygymrepo')

import numpy as np
import argparse
import slimevolleygym 
from slimevolleygym import BaselinePolicy

from agents_wraps.ppo2 import PPO_TEAM
from agents_wraps.ppo_roles import ROLES_TEAM

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
        self.opponent = None
        self.renderMode = renderMode
        self.survival_bonus = False

    def set_opponent(self, opponent):
        self.opponent = opponent

    def predict(self, obs1, obs2): # Opponent policy
        if self.opponent == None:
            print("Please set the opponent first")
            return

        action1, action2 = self.opponent.select_action(obs1, obs2)
        return action1, action2

    def step(self, action_1, action_2):
        if self.renderMode:
            self.render()
        return super(SlimeVolleyEvalEnv, self).step(action_1, action_2)


def rollout(env, policy_1, policy_2, render_mode=False):
  """ play one agent vs the other in modified gym-style loop. """
  obs_1, obs_2 = env.reset()

  done = False
  total_reward = 0

  while not done:

    action_1, action_2 = policy_1.predict(obs_1, obs_2)
    obs_arr, reward, done, _ = env.step(action_1, action_2)
    obs_1 = obs_arr[0]
    obs_2 = obs_arr[1]

    total_reward += reward

    if not policy_1.agent1 == None and hasattr(policy_1.agent1, 'roles'):
        policy_1.decide_role(obs_1, obs_2)
    
    if not policy_2.agent1 == None and hasattr(policy_2.agent1, 'roles'):
        policy_2.decide_role(obs_1, obs_2)

  return total_reward

def evaluate_multiagent(env, policy_1, policy_2, n_trials=1000, init_seed=721):
    print("2v2 Slimevolley Evaluation")
    history = []
    for i in range(1, n_trials+1):
        env.seed(seed=init_seed+i)
        cumulative_score = rollout(env, policy_1, policy_2, render_mode=render_mode)
        print("cumulative score #", i, ":", cumulative_score)
        history.append(cumulative_score)


    return history

if __name__=="__main__":

    APPROVED_MODELS = ["baseline", "ppo", "ppo_ad"]

    def check_choice(choice):
        choice = choice.lower()
        if choice not in APPROVED_MODELS:
            return False
        return True

    MODEL = {
        "baseline": MultiAgentBaselinePolicy,
        "ppo": PPO_TEAM,
        "ppo_ad": ROLES_TEAM
    }

    parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
    parser.add_argument('--left', help='choice of (baseline, ppo, ...)', type=str, default="baseline")
    parser.add_argument('--right', help='choice of (baseline, ppo, ...)', type=str, default="ppo")
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

    history = evaluate_multiagent(env, policy_1, policy_2)

    print("history dump:", history)
    print(right_agent_choice + " scored", np.round(np.mean(history), 3), "±", np.round(np.std(history), 3), "vs",
        left_agent_choice, "over")
