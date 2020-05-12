import torch
import numpy as np
from misc import onehot_from_logits
from agent import DDPGAgent
from utils import MultiAgentReplayBuffer
from vars import *
import os
import pickle as pkl


class MADDPG:

    def __init__(self, env, buffer_maxlen, model_name, discrete = False, greedy = False):
        self.env = env
        self.num_agents = env.n_agents
        self.replay_buffer = MultiAgentReplayBuffer(self.num_agents, buffer_maxlen)
        self.agents = [DDPGAgent(self.env, i, greedy = greedy, discrete=discrete) for i in range(self.num_agents)]
        self.model_name = model_name
        self.episode_done = 0
        self.episodes_before_train = EPISODES_BEFORE_TRAIN
        self.steps_done = 0
        self.discrete_actions = discrete
        self.greedy = greedy


    def get_actions(self, states, explore=False):
        actions = []
        for i in range(self.num_agents):
            action = self.agents[i].get_action(states[0][i], explore)
            actions.append(action)
        return actions

    def update(self, batch_size):
        if self.episode_done <= self.episodes_before_train:
            return

        obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, \
        global_state_batch, global_actions_batch, global_next_state_batch, done_batch = self.replay_buffer.sample(
            batch_size)

        for i in range(self.num_agents):
            obs_batch_i = obs_batch[i]
            indiv_action_batch_i = indiv_action_batch[i]
            indiv_reward_batch_i = indiv_reward_batch[i]
            next_obs_batch_i = next_obs_batch[i]

            next_global_actions = []
            for agent in self.agents:
                next_obs_batch_i = torch.tensor(next_obs_batch_i, device=device, dtype=torch.float)
                indiv_next_action = agent.actor_target.forward(next_obs_batch_i)

                if self.discrete_actions:

                    indiv_next_action = [onehot_from_logits(indiv_next_action_j) for indiv_next_action_j in
                                     indiv_next_action]

                    indiv_next_action = torch.stack(indiv_next_action)
                next_global_actions.append(indiv_next_action)
            next_global_actions = torch.cat([next_actions_i for next_actions_i in next_global_actions], 1)


            self.agents[i].update(indiv_reward_batch_i, obs_batch_i, global_state_batch, global_actions_batch,
                                  global_next_state_batch, next_global_actions)
            if self.steps_done % TARGET_UPDATE == 0:
                self.agents[i].target_update()

    def normalize_states(self, states):
        normalized_states = []
        for i in range(self.num_agents):
            normalized_states.append(self.agents[i].normalize(states[i]))
        return np.array(normalized_states)

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def run(self, max_episode, max_steps, batch_size):
        episode_rewards = []
        for episode in range(max_episode):
            explr_pct_remaining = max(0, n_exploration_eps - episode) / n_exploration_eps
            self.scale_noise(
                final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)
            self.reset_noise()
            states = self.env.reset()
            states = self.normalize_states(states)
            states = torch.tensor(states, dtype=torch.float, device=device).unsqueeze(0)
            episode_reward = 0
            for step in range(max_steps):
                actions = self.get_actions(states, explore=True)
                next_states, rewards, dones = self.env.step(actions)
                next_states = self.normalize_states(next_states)
                next_states = torch.tensor(next_states, dtype=torch.float, device=device).unsqueeze(0)
                episode_reward += np.mean(rewards)

                if all(dones) or step == max_steps - 1:
                    dones = [1 for _ in range(self.num_agents)]
                    self.replay_buffer.push(states, actions, rewards, next_states, dones)
                    episode_rewards.append(episode_reward)
                    print("episode: {}  |  reward: {}  \n".format(episode, np.round(episode_reward, decimals=4)))
                    break
                else:
                    dones = [0 for _ in range(self.num_agents)]
                    self.replay_buffer.push(states, actions, rewards, next_states, dones)
                    states = next_states

                    if len(self.replay_buffer) > batch_size:
                        self.update(batch_size)
                self.steps_done += 1
            self.episode_done += 1

            with open(os.getcwd() + '/data/output/' + self.model_name + '_rewards.txt', 'a') as f:
                f.write('Episode {}, Reward {} \n'.format(self.episode_done, episode_reward))

            if self.episode_done % 100 == 0:
                torch.save(self, os.getcwd() + '/data/output/' + self.model_name + '_intermediate_model.pt')

        model_params = {'NUM_EPISODES': NUM_EPISODES,
                        'EPSILON': EPSILON,
                        'EPS_DECAY': EPS_DECAY,
                        'LEARNING_RATE_ACTOR': LEARNING_RATE_ACTOR,
                        'LEARNING_RATE_CRITIC': LEARNING_RATE_CRITIC,
                        'GAMMA': GAMMA,
                        'TARGET_UPDATE': TARGET_UPDATE,
                        'BATCH_SIZE': BATCH_SIZE,
                        'TIME_STEP_SIZE': TIME_STEP_SIZE,
                        'NUM_HOURS': NUM_HOURS,
                        'COMFORT_PENALTY': COMFORT_PENALTY,
                        'MEMORY_SIZE': MEMORY_SIZE}

        episode_rewards.append(model_params)
        with open(os.getcwd() + '/data/output/' + self.model_name + '_rewards_maddpg.pkl',
                  'wb') as f:
            pkl.dump(episode_rewards, f)