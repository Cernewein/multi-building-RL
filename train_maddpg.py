import torch
import numpy as np
from agent import DDPGAgent
from MADDPG import MADDPG
from utils import MultiAgentReplayBuffer
from MultiBuildingEnv import env
from vars import *
import os
import pickle as pkl


def train_maddpg(ckpt,model_name, discrete):
    environment = env(discrete=discrete)

    ma_controller = MADDPG(environment, MEMORY_SIZE, model_name, discrete)
    ma_controller.run(NUM_EPISODES,NUM_TIME_STEPS,BATCH_SIZE)

    torch.save(ma_controller, os.getcwd() + '/data/output/' + model_name + 'model.pt')