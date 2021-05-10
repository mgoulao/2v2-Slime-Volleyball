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

import gym
import os
import numpy as np
import argparse
import slimevolleygym 
from slimevolleygym import BaselinePolicy
from time import sleep

from agents_wraps.ppo2 import PPO_TEAM

class MultiAgentBaselinePolicy(BaselinePolicy):
    def __init__(self, env):
        super(MultiAgentBaselinePolicy, self).__init__()

    def predict(self, obs_1, obs_2):
        return self.predict(obs_1), self.predict(obs_2)

class SlimeVolleyEvalEnv(slimevolleygym.SlimeVolleyEnv):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, renderMode):
        super(SlimeVolleyEvalEnv, self).__init__()
        self.opponent = None
        self.renderMode = renderMode

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
    obs_arr, reward, done, info = env.step(action, action1)
    obs_1 = obs_arr[0]
    obs_2 = obs_arr[1]

    total_reward += reward

  return total_reward

def evaluate_multiagent(env, policy_1, policy_2, n_trials=1000, init_seed=721):
    print("2v2 Slimevolley Evaluation")
    history = []
    for i in range(n_trials):
        env.seed(seed=init_seed+i)
        cumulative_score = rollout(env, policy_1, policy_2, render_mode=render_mode)
        print("cumulative score #", i, ":", cumulative_score)
        history.append(cumulative_score)
    return history

if __name__=="__main__":

    APPROVED_MODELS = ["baseline", "ppo"]

    def check_choice(choice):
        choice = choice.lower()
        if choice not in APPROVED_MODELS:
            return False
        return True

    MODEL = {
        "baseline": MultiAgentBaselinePolicy,
        "ppo": PPO_TEAM
    }

    parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
    parser.add_argument('--left', help='choice of (baseline, ppo, ...)', type=str, default="baseline")
    parser.add_argument('--right', help='choice of (baseline, ppo, ...)', type=str, default="ppo")
    parser.add_argument('--render', action='store_true', help='render to screen?', default=False)

    args = parser.parse_args()
    render_mode = args.render
    
    right_agent_choice = args.right
    left_agent_choice = args.left
    
    env = SlimeVolleyEvalEnv(render_mode)

    policy_1 = MODEL[right_agent_choice](env) # the right agent
    policy_2 = MODEL[left_agent_choice](env) # the left agent

    env.set_opponent(policy_2)

    assert check_choice(args.right), "Please enter a valid agent"
    assert check_choice(args.left), "Please enter a valid agent"

    history = evaluate_multiagent(env, policy_1, policy_2)

    print("history dump:", history)
    print(right_agent_choice+" scored", np.round(np.mean(history), 3), "±", np.round(np.std(history), 3), "vs",
        left_agent_choice, "over")
