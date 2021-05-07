import sys
sys.path.append('../slimevolleygymrepo')

import os
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import slimevolleygym 
from shutil import copyfile
import numpy as np

from agents_wraps.ppo2 import PPO_TEAM

BEST_THRESHOLD = 0.5

class SlimeVolleySelfPlayEnv(slimevolleygym.SlimeVolleyEnv):
  # wrapper over the normal single player env, but loads the best self play model
  def __init__(self, logdir, renderMode, selfplay):
    super(SlimeVolleySelfPlayEnv, self).__init__()
    self.logdir = logdir
    self.selfplay = selfplay
    self.selfplay_opponent = None
    self.best_model_filename = "best_model"
    self.renderMode = renderMode

  def predict(self, obs1, obs2): # the policy
    if self.selfplay_opponent is None:
      return self.action_space.sample(), self.action_space.sample() # return a random action
    else:
      action1, action2 = self.selfplay_opponent.select_action(obs1, obs2)
      return action1, action2

  def reset(self):
    # modellist = [f for f in os.listdir(self.logdir) if f.startswith("best_model_")]
    # if len(modellist) > 0 and self.selfplay_opponent is None and self.selfplay:
    #   print("SELFPLAY: Best Model Found!")
    #   self.load_opponent_best_model()
      
    return super(SlimeVolleySelfPlayEnv, self).reset()

  def load_opponent_best_model(self):
    self.policy = self
    if self.selfplay_opponent is not None:
      del self.selfplay_opponent
    self.selfplay_opponent = PPO_TEAM(self, self.logdir)
    self.selfplay_opponent.load(self.best_model_filename)
    print("Best Model Loaded!")
  
  def step(self, action_1, action_2):
    if self.renderMode:
      self.render()
    return super(SlimeVolleySelfPlayEnv, self).step(action_1, action_2)

  def checkpoint(self, agent, mean_reward):
    #print("Evaluate Checkpoint: ", mean_reward)
    if mean_reward > BEST_THRESHOLD and self.selfplay:
      print("--------------------------------------------------------------------------------------------")
      print("SELFPLAY: mean_reward achieved:", mean_reward)
      agent.save("best_model")
      self.load_opponent_best_model()
      print("--------------------------------------------------------------------------------------------")


# class SelfPlayCallback(EvalCallback):
#   # hacked it to only save new version of best model if beats prev self by BEST_THRESHOLD score
#   # after saving model, resets the best score to be BEST_THRESHOLD
#   def __init__(self, *args, **kwargs):
#     super(SelfPlayCallback, self).__init__(*args, **kwargs)
#     self.best_model_save_path = kwargs.get('best_model_save_path', './logs')
#     self.log_dir = kwargs.get("log_path", './logs')
#     self.best_mean_reward = BEST_THRESHOLD
#     self.generation = 0
    
#   def _on_step(self) -> bool:
#     result = super(SelfPlayCallback, self)._on_step()

#     # episode_rewards, episode_lengths = evaluate_policy(
#     #             self.model,
#     #             self.eval_env,
#     #             n_eval_episodes=self.n_eval_episodes,
#     #             render=self.render,
#     #             deterministic=self.deterministic,
#     #             return_episode_rewards=True,
#     #             warn=self.warn,
#     #             callback=self._log_success_callback,
#     #         )
#     # print(np.mean(episode_rewards))

#     if result and self.best_mean_reward > BEST_THRESHOLD:
#       self.generation += 1
#       print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
#       print("SELFPLAY: new best model, bumping up generation to", self.generation)
#       source_file = os.path.join(self.best_model_save_path, "best_model.zip")
#       backup_file = os.path.join(self.best_model_save_path, "history_"+str(self.generation).zfill(8)+".zip")
#       copyfile(source_file, backup_file)
#       self.best_mean_reward = BEST_THRESHOLD
#     return result
