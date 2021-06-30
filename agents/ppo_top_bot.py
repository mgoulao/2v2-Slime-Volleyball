# From https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

from agents.baseppo import BaseTeam, BasePPO

from roles import Bottom, Top


################################## PPO Policy ##################################


class PPO_TOP_BOT(BasePPO):
    def __init__(self, state_dim, action_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip, attacker=False):
        super().__init__(state_dim, action_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
        self.teammate = None
        self.role = Top() if attacker else Bottom()

    def reward(self, prev_state, state, reward):
        return self.role.reward(prev_state, state, reward)

    def decide(self, state):
        self.role.decide(self, state, self.teammate)

################################## Top Bot Team ##################################

class TOP_BOT_TEAM(BaseTeam):

    logdir = "./top_bot_saves"
    logs  = "logs/top_bot_1"

    def __init__(self, env, logdir=None):
        super().__init__(env, logdir)
      
        self.agent1 = PPO_TOP_BOT(self.state_dim, self.action_space, self.lr_actor, self.lr_critic, self.gamma, self.K_epochs, self.eps_clip, attacker=True)
        self.agent2 = PPO_TOP_BOT(self.state_dim, self.action_space, self.lr_actor, self.lr_critic, self.gamma, self.K_epochs, self.eps_clip, attacker=False)
        self.agent1.teammate = self.agent2
        self.agent2.teammate = self.agent1

    def reward(self, prev_state_1, prev_state_2, state_1, state_2, reward):
        return self.agent1.reward(prev_state_1, state_1, reward), \
            self.agent2.reward(prev_state_2, state_2, reward)

    def decide_role(self, state_1, state_2):
        return self.agent1.decide(state_1), \
            self.agent2.decide(state_2)

    def train(self, total_timesteps):
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

        role_decide_freq = 5

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

                reward_1, reward_2 = self.reward(state_1, state_2, state_arr[0], state_arr[1], reward)

                state_1 = state_arr[0]
                state_2 = state_arr[1]

                # saving reward and is_terminals
                self.agent1.buffer.rewards.append(reward_1)
                self.agent1.buffer.is_terminals.append(done)
                self.agent2.buffer.rewards.append(reward_2)
                self.agent2.buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward

                if time_step % role_decide_freq == 0:
                    self.decide_role(state_1, state_2)

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
    
    @staticmethod
    def bestSaveExists():
        return BaseTeam.existsBestModel(TOP_BOT_TEAM.logdir)