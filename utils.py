import torch
from vars import *
from collections import namedtuple, deque
import random
from environment import Building
import numpy as np

# Taken from
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
import os
import pickle as pkl

class Normalizer():
    """
    Normalizes the input data by computing an online variance and mean
    """
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs).to(device)
        self.mean = torch.zeros(num_inputs).to(device)
        self.mean_diff = torch.zeros(num_inputs).to(device)
        self.var = torch.zeros(num_inputs).to(device)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var).to(device)
        return (inputs - self.mean)/obs_std


class ReplayMemory(object):
    """
    This class serves as storage capability for the replay memory. It stores the Transition tuple
    (state, action, next_state, reward) that can later be used by a DQN agent for learning based on experience replay.

    :param capacity: The size of the replay memory
    :type capacity: Integer
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition. (the transition tuple)"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Randomly selects batch_size elements from the memory.

        :param batch_size: The wanted batch size
        :type batch_size: Integer
        :return:
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class MultiAgentReplayBuffer:

    def __init__(self, num_agents, max_size):
        self.max_size = max_size
        self.num_agents = num_agents
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array(reward), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        obs_batch = [[] for _ in range(self.num_agents)]  # [ [states of agent 1], ... ,[states of agent n] ]    ]
        indiv_action_batch = [[] for _ in range(self.num_agents)]  # [ [actions of agent 1], ... , [actions of agent n]]
        indiv_reward_batch = [[] for _ in range(self.num_agents)]
        next_obs_batch = [[] for _ in range(self.num_agents)]

        global_state_batch = []
        global_next_state_batch = []
        global_actions_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)


        for experience in batch:
            state, action, reward, next_state, done = experience

            for i in range(self.num_agents):
                obs_i = state[0][i].detach().numpy()
                action_i = action[i]
                reward_i = reward[i]
                next_obs_i = next_state[0][i].detach().numpy()

                obs_batch[i].append(obs_i)
                indiv_action_batch[i].append(action_i)
                indiv_reward_batch[i].append(reward_i)
                next_obs_batch[i].append(next_obs_i)


            global_state_batch.append(np.concatenate(state.detach().numpy()))
            global_actions_batch.append(torch.cat(action))
            global_next_state_batch.append(np.concatenate(next_state.detach().numpy()))
            done_batch.append(done)

        return obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, global_state_batch, global_actions_batch, global_next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
