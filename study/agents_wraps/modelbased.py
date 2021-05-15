# From https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

import math

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import os

################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")



################################## PPO Policy ##################################

class ModelBuffer:
    def __init__(self):
        self.state_stack = []
        self.next_state = []

    def clear(self):
        del self.state_stack[:]
        del self.next_state[:]

    def sample():
        pass

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # I get two prev states and one action, you get a next state and reward
        self.lr = 0.01
        self.model = nn.Sequential(
                        nn.Linear(state_dim*2 + action_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, state_dim + 1)
                    )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        

    def forward(self, x):
        return self.model(x)

    def update(self, y_, y):
        self.zero_grad()

        loss = self.criterion(y_, y)
        loss.backward()
        self.optimizer.step()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1)
                    )

        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

    def forward(self):
        raise NotImplementedError


    def act(self, state):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()


    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip):

        self.state_dim = state_dim
        self.action_space = action_space
        self.action_dim = 2**action_space  # we want all the possible combinations of actions

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, self.action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.model_buffer = ModelBuffer()
        self.model = Model(state_dim)

        self.prev_states = []

    def select_action(self, state, env=True):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        
        if env:
            if len(self.prev_state) == 2:
                self.model_buffer.state_stack.append(torch.stack(self.prev_states))
                self.model_buffer.next_state.append(torch.stack(state))
                self.prev_states = self.prev_states[1:]

            self.prev_states.append(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        if env:
            return self.convert_action(action.item())
        else:
            return action.item()

    def convert_action(self, action):
        action = int(bin(action)[2:])
        env_action = np.zeros(self.action_space)
        i = 0
        while action:
            env_action[-1-i] = action % 10
            action = action//10
            i += 1
        return env_action

    def interact_model(self):
        y = []
        y_ = []

        for i in range(10):
            state_stack, next_state, reward = self.model_buffer.sample()
            action = self.select_action(state_stack[-1], env=False)

            # Predict
            prediction = self.model(torch.cat((state_stack, torch.FloatTensor(action))))
            reward_prediction = prediction[:self.state_dim]

            #Interact
            self.buffer.rewards.append(reward_prediction)
            self.buffer.is_terminals.append(False)

            y_.append(prediction)
            y.append(torch.cat((next_state, torch.FloatTensor(reward))))

        return y_, y

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Interact and Update Model
        y_, y = self.interact_model()
        self.model.update(y_, y)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)


    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
    
class Model_Team:

    def __init__(self, env, logdir="./model_saves"):
        self.logdir = logdir
        
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self.env = env

        state_dim = env.observation_space.shape[0]
        action_space = env.action_space.shape[0] 
        K_epochs = 10
        eps_clip = 0.2
        gamma = 0.99
        lr_actor = 0.0003
        lr_critic = 0.001 
        self.agent1 = PPO(state_dim, action_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip, False)
        self.agent2 = PPO(state_dim, action_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip, False)
        self.writer  = SummaryWriter('logs/ppo_1')

    def select_action(self, state1, state2):
        return self.agent1.select_action(state1), self.agent2.select_action(state2)

    def predict(self, state1, state2):
        return self.select_action(state1, state2)

    def train(self, total_timesteps):
        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        max_ep_len = 1000
        update_timestep = 4096 
        log_freq = total_timesteps / 100 
        print_freq = total_timesteps / 10
        checkpoint_model_freq = 1000

        t = 1
        # training loop
        while time_step < total_timesteps:

            state_1, state_2 = self.env.reset()
            current_ep_reward = 0
            done = False
            while not done:
                # select action with policy
                action_1, action_2 = self.select_action(state_1, state_2)
                state_arr, reward, done, _ = self.env.step(action_1, action_2)
                state_1 = state_arr[0]
                state_2 = state_arr[1]

                # saving reward and is_terminals
                self.agent1.model_buffer.rewards.append(reward)
                self.agent2.model_buffer.rewards.append(reward)

                self.agent1.buffer.rewards.append(reward)
                self.agent1.buffer.is_terminals.append(done)
                self.agent2.buffer.rewards.append(reward)
                self.agent2.buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % update_timestep == 0:
                    self.agent1.update()
                    self.agent2.update()

                # log in logging file
                if time_step % log_freq == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    self.writer.add_scalar('training reward', log_avg_reward, time_step)

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % print_freq == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("{}\% - Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format((time_step/total_timesteps)*100,\
                         i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % checkpoint_model_freq == 0:
                    self.env.checkpoint(self, current_ep_reward/t)
                # break; if the episode is over
                if done:
                    break
                
                t += 1
                    
            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        self.env.close()
       
    def loadBestModel(self):
        bestSaveExists = os.path.exists(f'{self.logdir}/best_model_agent1')
        if bestSaveExists:
            print("TEAM 1: Best Model Loaded!")
            self.load("best_model")

    def saveBestModel(self):
        self.save("best_model")

    def save(self, filename):
        self.agent1.save(f'{self.logdir}/{filename}_agent1')
        self.agent2.save(f'{self.logdir}/{filename}_agent2')

    def load(self, filename):
        self.agent1.load(f'{self.logdir}/{filename}_agent1')
        self.agent2.load(f'{self.logdir}/{filename}_agent2')
