import sys
sys.path.append('../slimevolleygymrepo')

import os
from stable_baselines3.common.callbacks import EvalCallback

import slimevolleygym 

BEST_THRESHOLD = 0.5 # must achieve a mean score above this to replace prev best self


class SlimeVolleySelfPlayEnv(slimevolleygym.SlimeVolleyEnv):
  # wrapper over the normal single player env, but loads the best self play model
  def __init__(self, logdir):
    super(SlimeVolleySelfPlayEnv, self).__init__()
    self.logdir = logdir
    self.policy = self
    self.best_model = None
    self.best_model_filename = None
  def predict(self, obs): # the policy
    if self.best_model is None:
      return self.action_space.sample() # return a random action
    else:
      action, _ = self.best_model.predict(obs)
      return action
  def reset(self):
    # load model if it's there
    modellist = [f for f in os.listdir(self.logdir) if f.startswith("history")]
    modellist.sort()
    if len(modellist) > 0:
      filename = os.path.join(self.logdir, modellist[-1]) # the latest best model
      if filename != self.best_model_filename:
        print("loading model: ", filename)
        self.best_model_filename = filename
        if self.best_model is not None:
          del self.best_model
        self.best_model = PPO1.load(filename, env=self)
    return super(SlimeVolleySelfPlayEnv, self).reset()

class SelfPlayCallback(EvalCallback):
  # hacked it to only save new version of best model if beats prev self by BEST_THRESHOLD score
  # after saving model, resets the best score to be BEST_THRESHOLD
  def __init__(self, *args, **kwargs):
    super(SelfPlayCallback, self).__init__(*args, **kwargs)
    self.best_model_save_path = kwargs.get('best_model_save_path', './logs')
    self.log_path = kwargs.get("log_path", './logs')
    self.best_mean_reward = BEST_THRESHOLD
    self.generation = 0
  def _on_step(self) -> bool:
    result = super(SelfPlayCallback, self)._on_step()
    if result and self.best_mean_reward > BEST_THRESHOLD:
      self.generation += 1
      print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
      print("SELFPLAY: new best model, bumping up generation to", self.generation)
      source_file = os.path.join(self.best_model_save_path, "best_model.zip")
      backup_file = os.path.join(self.best_model_save_path, "history_"+str(self.generation).zfill(8)+".zip")
      copyfile(source_file, backup_file)
      self.best_mean_reward = BEST_THRESHOLD
    return result
