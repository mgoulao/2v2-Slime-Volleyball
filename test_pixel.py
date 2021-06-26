"""
Human vs AI in pixel observation environment

Note that for multiagent mode, otherObs's image is horizontally flipped

Performance, 100,000 frames in 144.839 seconds, or 690 fps.
"""

import gym
import slimevolleygym
from time import sleep
from pyglet.window import key

from gym.envs.classic_control import rendering as rendering # to show actual obs2

if __name__=="__main__":
  """
  Example of how to use Gym env, in single or multiplayer setting

  Humans can override controls:

  Humans can override controls:

  blue Agent:
  W - Jump
  A - Left
  D - Right

  purple Agent:
  F - Jump
  C - Left
  B - Right
 
  yellow Agent:
  Up Arrow, Left Arrow, Right Arrow

  orange Agent:
  I - Jump
  J - Left
  L - Right
  """

  manualAction1 = [0, 0, 0] # forward, backward, jump
  manualAction2 = [0, 0, 0]
  manualAction3 = [0, 0, 0]
  manualAction4 = [0, 0, 0]
  manualMode1 = False 
  manualMode2 = False 
  manualMode3 = False
  manualMode4 = False 
  
  # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
  def key_press(k, mod):
    global manualMode1, manualMode2, manualMode3, manualMode4, manualAction1, manualAction2, manualAction3, manualAction4
    if k == key.LEFT:  manualAction1[0] = 1
    if k == key.RIGHT: manualAction1[1] = 1
    if k == key.UP:    manualAction1[2] = 1
    if (k == key.LEFT or k == key.RIGHT or k == key.UP): manualMode1 = True

    if k == key.J:  manualAction2[0] = 1
    if k == key.L:  manualAction2[1] = 1
    if k == key.I:  manualAction2[2] = 1
    if (k == key.J or k == key.L or k == key.I): manualMode2 = True

    if k == key.D:  manualAction3[0] = 1
    if k == key.A:  manualAction3[1] = 1
    if k == key.W:  manualAction3[2] = 1
    if (k == key.D or k == key.A or k == key.W): manualMode3 = True

    if k == key.B:  manualAction4[0] = 1
    if k == key.C:  manualAction4[1] = 1
    if k == key.F:  manualAction4[2] = 1
    if (k == key.B or k == key.C or k == key.F):manualMode4 = True

  def key_release(k, mod):
    global manualMode1, manualMode2, manualMode3, manualMode4, manualAction1, manualAction2, manualAction3, manualAction4
    if k == key.LEFT:  manualAction1[0] = 0
    if k == key.RIGHT: manualAction1[1] = 0
    if k == key.UP:    manualAction1[2] = 0
    if k == key.J:     manualAction2[0] = 0
    if k == key.L:     manualAction2[1] = 0
    if k == key.I:     manualAction2[2] = 0
    if k == key.D:     manualAction3[0] = 0
    if k == key.A:     manualAction3[1] = 0
    if k == key.W:     manualAction3[2] = 0
    if k == key.B:     manualAction4[0] = 0
    if k == key.C:     manualAction4[1] = 0
    if k == key.F:     manualAction4[2] = 0

  viewer = rendering.SimpleImageViewer(maxwidth=2160)

  env = gym.make("MultiAgentSlimeVolleySurvivalNoFrameskip-v0")
  policy = slimevolleygym.BaselinePolicy() # throw in a default policy (based on state, not pixels)

  obs = env.reset()
  env.render()

  env.viewer.window.on_key_press = key_press
  env.viewer.window.on_key_release = key_release

  defaultAction = [0, 0, 0]
  action1 = [0, 0, 0]
  action2 = [0, 0, 0]
  action3 = [0, 0, 0]
  action4 = [0, 0, 0]

  for t in range(10000):
    if manualMode1: # override with keyboard
      action1 = manualAction1
    else:
      action1 = defaultAction
    
    if manualMode2: # override with keyboard
      action2 = manualAction2
    else:
      action2 = defaultAction

    if manualMode3:
      action3 = manualAction3
      
    if manualMode4:
      action4 = manualAction4
    obs, reward, done, info = env.step(action1, action2, action3, action4) 

    otherObs = info['otherObs']

    state = info['state'] # cheat and look at the actual state (to find default actions quickly)
    defaultAction = policy.predict(state)
    sleep(0.02)
    #viewer.imshow(otherObs) # show the opponent's observtion (horizontally flipped)
    env.render()
    if done:
      obs = env.reset()
    if (t+1) % 5000 == 0:
      print(t+1)

  viewer.close()
  env.close()
