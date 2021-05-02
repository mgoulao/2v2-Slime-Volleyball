"""
State mode (Optional Human vs Built-in AI)

FPS (no-render): 100000 steps /7.956 seconds. 12.5K/s.
"""

import math
import numpy as np
import gym
import slimevolleygym

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER_MODE = True


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

  if RENDER_MODE:
    from pyglet.window import key
    from time import sleep

  manualAction1 = [0, 0, 0] # forward, backward, jump
  manualAction2 = [0, 0, 0]
  manualAction3 = [0, 0, 0]
  manualAction4 = [0, 0, 0]
  manualMode1 = True #change back to False 
  manualMode2 = True #change back to False 
  manualMode3 = True #change back to False
  manualMode4 = True #change back to False 
  
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

  policy = slimevolleygym.BaselinePolicy() # defaults to use RNN Baseline for player

  env = gym.make("SlimeVolley-v0")
  env.seed(np.random.randint(0, 10000))
  #env.seed(689)

  if RENDER_MODE:
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

  obs = env.reset()

  steps = 0
  total_reward = 0
  action1 = np.array([0, 0, 0])
  action2 = np.array([0, 0, 0])
  action3 = np.array([0, 0, 0])
  action4 = np.array([0, 0, 0])

  done = False

  while not done:

    if manualMode1: # override with keyboard
      action1 = manualAction1
    else:
      action1 = policy.predict(obs)

    if manualMode2:
      action2 = manualAction2
      obs, reward, done, _ = env.step(action1, action2, action3, action4)
    else:
      obs, reward, done, _ = env.step(action1)
    if manualMode3:
      action3 = manualAction3
      obs, reward, done, _ = env.step(action1, action2, action3, action4)
    else:
      obs, reward, done, _ = env.step(action1)
    if manualMode4:
      action4 = manualAction4
      obs, reward, done, _ = env.step(action1, action2, action3, action4)
    else:
      obs, reward, done, _ = env.step(action1)

    if reward > 0 or reward < 0:
      manualMode1 = True #change back to False
      manualMode2 = True #change back to False
      manualMode3 = True #change back to False
      manualMode4 = True #change back to False

    total_reward += reward

    if RENDER_MODE:
      env.render()
      sleep(0.02) # 0.01

  env.close()
  print("cumulative score", total_reward)
