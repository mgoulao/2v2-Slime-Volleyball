import sys

sys.path.append('../slimevolleygymrepo')

import slimevolleygym 

BEST_THRESHOLD = 8
class SlimeVolleySelfPlayEnv(slimevolleygym.SlimeVolleyEnv):
  # wrapper over the normal single player env, but loads the best self play model
  def __init__(self, team, renderMode, selfplay):
    super(SlimeVolleySelfPlayEnv, self).__init__()
    self.Team = team
    self.selfplay = selfplay
    if selfplay:
      self.policy = self
    self.selfplay_opponent = None
    self.renderMode = renderMode

  def predict(self, obs1, obs2): # the policy
    if self.selfplay_opponent is None:
      return self.action_space.sample(), self.action_space.sample() # return a random action
    else:
      action1, action2 = self.selfplay_opponent.select_action(obs1, obs2)
      return action1, action2

  def reset(self):
    bestSaveExists = self.Team.bestSaveExists()
    if bestSaveExists and self.selfplay_opponent is None and self.selfplay:
      print("SELFPLAY: Best Model Found!")
      self.load_opponent_best_model()
      
    return super(SlimeVolleySelfPlayEnv, self).reset()

  def load_opponent_best_model(self):
    self.policy = self
    if self.selfplay_opponent is not None:
      del self.selfplay_opponent
    self.selfplay_opponent = self.Team(self)
    print("SELFPLAY: ", end="")
    self.selfplay_opponent.loadBestModel()
  
  def step(self, action_1, action_2):
    if self.renderMode:
      self.render()
    return super(SlimeVolleySelfPlayEnv, self).step(action_1, action_2)

  def checkpoint(self, agent, mean_reward):
    if mean_reward > BEST_THRESHOLD and self.selfplay:
      print("--------------------------------------------------------------------------------------------")
      print("SELFPLAY: mean_reward achieved:", mean_reward)
      agent.saveBestModel()
      self.load_opponent_best_model()
      print("--------------------------------------------------------------------------------------------")
