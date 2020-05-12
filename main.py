from DQN import DAgent
import sys
from environment import *
from matplotlib import style
style.use('ggplot')
from vars import *
from itertools import count
import pickle as pkl
import os
import argparse
import sys
import torch
import pandas as pd
import numpy as np
from train_dqn import train_dqn
from train_maddpg import train_maddpg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--model_name", default='')
    parser.add_argument("--dynamic", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--soft", default=False,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--eval", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--model_type", default='MADDPG')
    parser.add_argument("--discrete", default=False, type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()


def run(ckpt,model_name,dynamic,soft, eval, model_type, discrete):

    if not eval:
        if model_type != 'MADDPG':
            train_dqn(ckpt, model_name, dynamic, soft)
        else:
            train_maddpg(ckpt, model_name, discrete = discrete)

    else:
        if ckpt:
            brain = torch.load(ckpt,map_location=torch.device('cpu'))
            brain.epsilon = 0
            brain.eps_end = 0
            env = System(eval=True)
            inside_temperatures_1 = [env.buildings[0].inside_temperature]
            inside_temperatures_2 = [env.buildings[1].inside_temperature]
            base_loads_1 = [env.buildings[0].base_load]
            base_loads_2 = [env.buildings[1].base_load]
            ambient_temperatures = [env.ambient_temperature]
            total_loads = [env.total_load]
            actions = [0]
            rewards=[0]
            print('Starting evaluation of the model')
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float).to(device)
            # Normalizing data using an online algo
            brain.normalizer.observe(state)
            state = brain.normalizer.normalize(state).unsqueeze(0)
            for t_episode in range(NUM_TIME_STEPS):
                action = brain.select_action(state).type(torch.FloatTensor)
                actions.append(action.item())
                next_state, reward, done = env.step(action.item())
                rewards.append(reward)
                inside_temperatures_1.append(env.buildings[0].inside_temperature)
                inside_temperatures_2.append(env.buildings[1].inside_temperature)
                base_loads_1.append(env.buildings[0].base_load)
                base_loads_2.append(env.buildings[1].base_load)
                ambient_temperatures.append(env.ambient_temperature)
                total_loads.append(env.total_load)
                if not done:
                    next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                    # normalize data using an online algo
                    brain.normalizer.observe(next_state)
                    next_state = brain.normalizer.normalize(next_state).unsqueeze(0)
                else:
                    next_state = None
                # Move to the next state
                state = next_state

            eval_data = pd.DataFrame()
            eval_data['Inside Temperatures 1'] = inside_temperatures_1
            eval_data['Inside Temperatures 2'] = inside_temperatures_2
            eval_data['Base Loads 1'] = base_loads_1
            eval_data['Base Loads 2'] = base_loads_2
            eval_data['Ambient Temperatures'] = ambient_temperatures
            eval_data['Actions'] = actions
            eval_data['Rewards'] = rewards
            eval_data['Total Load'] = total_loads
            with open(os.getcwd() + '/data/output/' + model_name + '_eval.pkl', 'wb') as f:
                pkl.dump(eval_data, f)


            # Evaluation if price was kept constant
            env = System(eval=True)
            inside_temperatures_1 = [env.buildings[0].inside_temperature]
            inside_temperatures_2 = [env.buildings[1].inside_temperature]
            base_loads_1 = [env.buildings[0].base_load]
            base_loads_2 = [env.buildings[1].base_load]
            ambient_temperatures = [env.ambient_temperature]
            total_loads = [env.total_load]
            rewards = [0]
            print('Starting evaluation of the model')
            state = env.reset()
            for t_episode in range(NUM_TIME_STEPS):
                next_state, reward, done = env.step(0)
                rewards.append(reward)
                inside_temperatures_1.append(env.buildings[0].inside_temperature)
                inside_temperatures_2.append(env.buildings[1].inside_temperature)
                base_loads_1.append(env.buildings[0].base_load)
                base_loads_2.append(env.buildings[1].base_load)
                ambient_temperatures.append(env.ambient_temperature)
                total_loads.append(env.total_load)
            eval_data = pd.DataFrame()
            eval_data['Inside Temperatures 1'] = inside_temperatures_1
            eval_data['Inside Temperatures 2'] = inside_temperatures_2
            eval_data['Base Loads 1'] = base_loads_1
            eval_data['Base Loads 2'] = base_loads_2
            eval_data['Ambient Temperatures'] = ambient_temperatures
            eval_data['Rewards'] = rewards
            eval_data['Total Load'] = total_loads
            with open(os.getcwd() + '/data/output/' + model_name + 'base_eval.pkl', 'wb') as f:
                pkl.dump(eval_data, f)

            print('Finished evaluation on January.')

if __name__ == '__main__':
    args = parse_args()
    run(**vars(args))