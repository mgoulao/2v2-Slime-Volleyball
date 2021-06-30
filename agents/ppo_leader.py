from agents.baseppo import BaseTeam, BasePPO, device

import torch
import numpy as np

################################## PPO Leader ##################################

class PPO_LEADER(BasePPO):
    def __init__(self, state_dim, action_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.action_dim = self.action_space**2
        super().__init__(state_dim, action_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    def convert_action(self, action):
        # env_action = np.zeros(self.action_space)
        # env_action[action%self.action_space] = 1
        # return env_action
        return action
        
################################## PPO Slave ##################################

class PPO_SLAVE(BasePPO):

    def __init__(self, state_dim, action_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        state_dim += 1
        self.action_dim = action_space        
        super().__init__(state_dim, action_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    ''' TODO: review this '''
    def predict(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.item()

    def convert_action(self, action):
        env_action = np.zeros(self.action_space)
        env_action[action%self.action_space] = 1
        return env_action

    
class LEADER_TEAM(BaseTeam):

    logdir = "./leader_saves"
    logs  = "logs/leader_1"

    def __init__(self, env, logdir=None):
        super().__init__(env, logdir)

        self.agent1 = PPO_LEADER(self.state_dim, self.action_space, self.lr_actor, self.lr_critic, self.gamma, self.K_epochs, self.eps_clip)
        self.agent2 = PPO_SLAVE(self.state_dim, self.action_space, self.lr_actor, self.lr_critic, self.gamma, self.K_epochs, self.eps_clip)

    def predict(self, state1, state2):
        action1 = self.agent1.predict(state1)
        state2 = np.append(state2, action1 // self.action_space)
        action2 = self.agent2.predict(state2)

        if action1 // self.action_space == self.agent2:
            self.curr_leader_reward = self.LEADER_REWARD 
        
        return self.agent1.convert_action(action1), self.agent2.convert_action(action2)

    def train(self, total_timesteps):
        self.agent1.training = True
        self.agent2.training = True

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        checkpoint_running_reward = 0
        checkpoint_running_episodes = 0

        time_step = 0
        i_episode = 0

        update_timestep = 4096 
        log_freq = total_timesteps / 100 
        print_freq = total_timesteps / 10
        checkpoint_model_freq = 10000

        t = 1
        # training loop
        while time_step < total_timesteps:

            state_1, state_2 = self.env.reset()
            current_ep_reward = 0
            done = False
            while not done:
                # select action with policy
                action_1, action_2 = self.predict(state_1, state_2)
                state_arr, reward, done, _ = self.env.step([action_1, action_2])
                state_1 = state_arr[0]
                state_2 = state_arr[1]

                # saving reward and is_terminals
                self.agent1.buffer.rewards.append(reward)
                self.agent1.buffer.is_terminals.append(done)
                self.agent2.buffer.rewards.append(reward+self.curr_leader_reward)
                self.agent2.buffer.is_terminals.append(done)
                self.curr_leader_reward = 0

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

                # Check for selfplay
                if time_step % checkpoint_model_freq == 0:
                    checkpoint_avg_reward = checkpoint_running_reward / checkpoint_running_episodes
                    checkpoint_avg_reward = round(checkpoint_avg_reward, 4)

                    self.env.checkpoint(self, checkpoint_avg_reward)

                    checkpoint_running_reward = 0
                    checkpoint_running_episodes = 0

                if done:
                    break
                
                t += 1
                    
            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            checkpoint_running_reward += current_ep_reward
            checkpoint_running_episodes += 1

            i_episode += 1

        self.env.close()
        self.agent1.training = False
        self.agent2.training = False
        
    @staticmethod
    def bestSaveExists():
        return BaseTeam.existsBestModel(LEADER_TEAM.logdir)