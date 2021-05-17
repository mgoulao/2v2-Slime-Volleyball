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
    obs_arr, reward, done, info = env.step(action_1, action_2)
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

<<<<<<< HEAD
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
    print(right_agent_choice+" scored", np.round(np.mean(history), 3), "±", np.round(np.std(history), 3), "vs",
        left_agent_choice, "over")
=======
  APPROVED_MODELS = ["baseline", "ppo", "ga", "cma", "random"]

  def checkchoice(choice):
    choice = choice.lower()
    if choice not in APPROVED_MODELS:
      return False
    return True

  PATH = {
    "baseline": None,
    "ppo": "zoo/ppo/best_model.zip",
    "cma": "zoo/cmaes/slimevolley.cma.64.96.best.json",
    "ga": "zoo/ga_sp/ga.json",
    "random": None,
  }

  MODEL = {
    "baseline": makeBaselinePolicy,
    "ppo": PPOPolicy,
    "cma": makeSlimePolicy,
    "ga": makeSlimePolicyLite,
    "random": RandomPolicy,
  }

  parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
  parser.add_argument('--left', help='choice of (baseline, ppo, cma, ga, random)', type=str, default="baseline")
  parser.add_argument('--leftpath', help='path to left model (leave blank for zoo)', type=str, default="")
  parser.add_argument('--right', help='choice of (baseline, ppo, cma, ga, random)', type=str, default="ga")
  parser.add_argument('--rightpath', help='path to right model (leave blank for zoo)', type=str, default="")
  parser.add_argument('--render', action='store_true', help='render to screen?', default=False)
  parser.add_argument('--day', action='store_true', help='daytime colors?', default=False)
  parser.add_argument('--pixel', action='store_true', help='pixel rendering effect? (note: not pixel obs mode)', default=False)
  parser.add_argument('--seed', help='random seed (integer)', type=int, default=721)
  parser.add_argument('--trials', help='number of trials (default 1000)', type=int, default=1000)

  args = parser.parse_args()

  if args.day:
    slimevolleygym.setDayColors()

  if args.pixel:
    slimevolleygym.setPixelObsMode()

  env = gym.make("SlimeVolley-v0")
  env.seed(args.seed)

  render_mode = args.render

  assert checkchoice(args.right), "pls enter a valid agent"
  assert checkchoice(args.left), "pls enter a valid agent"

  c0 = args.right
  c1 = args.left

  path0 = PATH[c0]
  path1 = PATH[c1]

  if len(args.rightpath) > 0:
    assert os.path.exists(args.rightpath), args.rightpath+" doesn't exist."
    path0 = args.rightpath
    print("path of right model", path0)

  if len(args.leftpath):
    assert os.path.exists(args.leftpath), args.leftpath+" doesn't exist."
    path1 = args.leftpath
    print("path of left model", path1)

  if c0.startswith("ppo") or c1.startswith("ppo"):
    from stable_baselines3 import PPO1

  policy0 = MODEL[c0](path0) # the right agent
  policy1 = MODEL[c1](path1) # the left agent

  history = evaluate_multiagent(env, policy0, policy1,
    render_mode=render_mode, n_trials=args.trials, init_seed=args.seed)

  print("history dump:", history)
  print(c0+" scored", np.round(np.mean(history), 3), "±", np.round(np.std(history), 3), "vs",
    c1, "over", args.trials, "trials.")
>>>>>>> origin/abstract_roles
