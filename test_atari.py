"""
Pixel observation environment (atari compatible example, w/ 84x84 resized 4-frame stack.
"""

import gym
from gym import spaces
import numpy as np
import slimevolleygym
from pyglet.window import key
from time import sleep
import cv2
from gym.envs.classic_control import rendering as rendering
from slimevolleygym import FrameStack, render_atari

class NoopResetEnv(gym.Wrapper):
  def __init__(self, env, noop_max=30):
    """
    (from stable-baselines)
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: (Gym Environment) the environment to wrap
    :param noop_max: (int) the maximum value of no-ops to run
    """
    gym.Wrapper.__init__(self, env)
    self.noop_max = noop_max
    self.override_num_noops = None
    self.noop_action = 0
    assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

  def reset(self, **kwargs):
    self.env.reset(**kwargs)
    if self.override_num_noops is not None:
      noops = self.override_num_noops
    else:
      noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
    assert noops > 0
    obs = None
    for _ in range(noops):
      obs, _, done, _ = self.env.step(self.noop_action, self.noop_action)
      if done:
        obs = self.env.reset(**kwargs)
    return obs

  def step(self, action):
      return self.env.step(action)

class MaxAndSkipEnv(gym.Wrapper):
  def __init__(self, env, skip=4):
    """
    (from stable baselines)
    Return only every `skip`-th frame (frameskipping)

    :param env: (Gym Environment) the environment
    :param skip: (int) number of `skip`-th frame
    """
    gym.Wrapper.__init__(self, env)
    # most recent raw observations (for max pooling across time steps)
    self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=env.observation_space.dtype)
    self._skip = skip

  def step(self, action):
    """
    Step the environment with the given action
    Repeat action, sum reward, and max over last observations.

    :param action: ([int] or [float]) the action
    :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
    """
    total_reward = 0.0
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if i == self._skip - 2:
        self._obs_buffer[0] = obs
      if i == self._skip - 1:
        self._obs_buffer[1] = obs
      total_reward += reward
      if done:
        break
    # Note that the observation on the done=True frame
    # doesn't matter
    max_frame = self._obs_buffer.max(axis=0)

    return max_frame, total_reward, done, info

  def reset(self, **kwargs):
      return self.env.reset(**kwargs)

class WarpFrame(gym.ObservationWrapper):
  def __init__(self, env):
    """
    (from stable-baselines)
    Warp frames to 84x84 as done in the Nature paper and later work.

    :param env: (Gym Environment) the environment
    """
    gym.ObservationWrapper.__init__(self, env)
    self.width = 84
    self.height = 84
    self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                        dtype=env.observation_space.dtype)

  def observation(self, frame):
    """
    returns the current observation from a frame

    :param frame: ([int] or [float]) environment frame
    :return: ([int] or [float]) the observation
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
    return frame[:, :, None]


def toAtariAction(action):
  """
  action_table = [[0, 0, 0], # NOOP
                  [1, 0, 0], # LEFT (forward)
                  [1, 0, 1], # UPLEFT (forward jump)
                  [0, 0, 1], # UP (jump)
                  [0, 1, 1], # UPRIGHT (backward jump)
                  [0, 1, 0]] # RIGHT (backward)
  """
  left = action[0]
  right = action[1]
  jump = action[2]
  if left == right:
    left = 0
    right = 0
  if left == 1 and jump == 0:
    return 1
  if left == 1 and jump == 1:
    return 2
  if right == 1 and jump == 0:
    return 5
  if right == 1 and jump == 1:
    return 4
  if jump == 1:
    return 3
  return 0

# simulate typical Atari Env:
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

  #env = gym.make("SlimeVolleyNoFrameskip-v0")
  env = slimevolleygym.SlimeVolleyAtariEnv()
  # typical Atari processing:
  env = NoopResetEnv(env, noop_max=30)
  env = MaxAndSkipEnv(env, skip=4)
  env = WarpFrame(env)
  env = FrameStack(env, 4)
  env.seed(689)

  obs = env.reset()

  for t in range(10000):

    if manualMode1: # override with keyboard
      action1 = manualAction1
    else:
      action1 = 0 
    
    if manualMode2: # override with keyboard
      action2 = manualAction2
    else:
      action2 = 0

    if manualMode3:
      action3 = manualAction3
    else:
      action3 = 0
      
    if manualMode4:
      action4 = manualAction4
    else:
      action4 = 0

    obs, reward, done, _ = env.step(action1, action2, action3, action4) 

    if reward > 0 or reward < 0:
      print("reward", reward)
      manualMode = False

    if reward > 0 or reward < 0:
      print(t, reward)

    render_img = render_atari(obs)
    viewer.imshow(render_img)
    sleep(0.08)

    if t == 0:
      viewer.window.on_key_press = key_press
      viewer.window.on_key_release = key_release

    if done:
      obs = env.reset()

  viewer.close()
  env.close()
