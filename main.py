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
from train_ddpg import train_ddpg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--model_name", default='')
    parser.add_argument("--dynamic", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--soft", default=False,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--eval", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--model_type", default='DDPG')
    parser.add_argument("--RL", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--january", default=False, type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()


def run(ckpt,model_name,dynamic,soft, eval, model_type, RL, january):

    if not eval:
        if model_type != 'DDPG':
            train_dqn(ckpt, model_name, dynamic, soft, RL)
        else:
            train_ddpg(model_name, RL)

    else:
        if ckpt:
            brain = torch.load(ckpt,map_location=torch.device('cpu'))
            brain.epsilon = 0
            brain.eps_end = 0

            # inside_temperatures_1 = []
            # inside_temperatures_2 = []
            # ambient_temperatures = []
            # times = []
            # base_loads = []
            # actions = []
            #
            # for inside_temp_1 in np.arange(18, 22, 1 / 5):
            #     print(inside_temp_1)
            #     for inside_temp_2 in np.arange(18, 22, 1 / 5):
            #         for ambient_temp in np.arange(-5, 5, 1 / 5):
            #             for load in np.arange(0, 30, 1/2):
            #                 for time in range(0,24):
            #                     scaled_load = load/1000
            #                     state = [ambient_temp, scaled_load, inside_temp_1, inside_temp_2, time]
            #                     state = torch.tensor(state, dtype=torch.float).to(device)
            #                     state = brain.normalizer.normalize(state).unsqueeze(0)
            #                     action = brain.select_action(state).type(torch.FloatTensor).item()
            #                     inside_temperatures_1.append(inside_temp_1)
            #                     inside_temperatures_2.append(inside_temp_2)
            #                     ambient_temperatures.append(ambient_temp)
            #                     times.append(time)
            #                     base_loads.append(load)
            #                     actions.append(action)
            #
            # eval_data = pd.DataFrame()
            # eval_data['Inside Temperatures 1'] = inside_temperatures_1
            # eval_data['Inside Temperatures 2'] = inside_temperatures_2
            # eval_data['Ambient Temperatures'] = ambient_temperatures
            # eval_data['Times'] = times
            # eval_data['Loads'] = base_loads
            # eval_data['Actions'] = actions
            # with open(os.getcwd() + '/data/output/' + model_name + 'policy_eval.pkl', 'wb') as f:
            #     pkl.dump(eval_data, f)

            env = System(eval=True, january = january, RL_building=RL)

            base_loads_1 = [env.buildings[0].base_load]
            base_loads_2 = [env.buildings[1].base_load]
            actions_building_1 = [0]
            actions_building_2 = [0]
            ambient_temperatures = [env.ambient_temperature]
            total_loads = [env.total_load]
            total_price = [env.total_power_cost]
            actions = []
            rewards = [0]
            print('Starting evaluation of the model')
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float).to(device)
            inside_temperatures_1 = [state[2].item()]
            inside_temperatures_2 = [state[3].item()]
            # Normalizing data using an online algo
            brain.normalizer.observe(state)
            state = brain.normalizer.normalize(state).unsqueeze(0)
            for t_episode in range(NUM_TIME_STEPS):
                action = brain.select_action(state).type(torch.FloatTensor)
                actions.append(action.item())
                next_state, reward, done = env.step(action.item())
                rewards.append(reward)
                inside_temperatures_1.append(next_state[2])
                inside_temperatures_2.append(next_state[3])
                actions_building_1.append(env.buildings[0].action)
                actions_building_2.append(env.buildings[1].action)
                base_loads_1.append(env.buildings[0].base_load)
                base_loads_2.append(env.buildings[1].base_load)
                ambient_temperatures.append(env.ambient_temperature)
                total_loads.append(env.total_load)
                total_price.append(env.total_power_cost)
                if not done:
                    next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                    # normalize data using an online algo
                    brain.normalizer.observe(next_state)
                    next_state = brain.normalizer.normalize(next_state).unsqueeze(0)
                else:
                    next_state = None
                # Move to the next state
                state = next_state
            actions.append(0)
            eval_data = pd.DataFrame()
            eval_data['Inside Temperatures 1'] = inside_temperatures_1
            eval_data['Inside Temperatures 2'] = inside_temperatures_2
            eval_data['Base Loads 1'] = base_loads_1
            eval_data['Base Loads 2'] = base_loads_2
            eval_data['Actions 1'] = actions_building_1
            eval_data['Actions 2'] = actions_building_2
            eval_data['Ambient Temperatures'] = ambient_temperatures
            eval_data['Actions'] = actions
            eval_data['Rewards'] = rewards
            eval_data['Total Load'] = total_loads
            eval_data['Total Price'] = total_price
            with open(os.getcwd() + '/data/output/' + model_name + '_eval.pkl', 'wb') as f:
                pkl.dump(eval_data, f)

            # Evaluation if price was kept constant
            #env = System(eval=True)
            state = env.reset()
            inside_temperatures_1 = [state[2].item()]
            inside_temperatures_2 = [state[3].item()]
            base_loads_1 = [env.buildings[0].base_load]
            base_loads_2 = [env.buildings[1].base_load]
            ambient_temperatures = [env.ambient_temperature]
            total_loads = [env.total_load]
            total_price = [env.total_power_cost]
            rewards = [0]
            print('Starting evaluation of the model')
            for t_episode in range(NUM_TIME_STEPS):
                next_state, reward, done = env.step(0)
                rewards.append(reward)
                inside_temperatures_1.append(next_state[2])
                inside_temperatures_2.append(next_state[3])
                base_loads_1.append(env.buildings[0].base_load)
                base_loads_2.append(env.buildings[1].base_load)
                ambient_temperatures.append(env.ambient_temperature)
                total_loads.append(env.total_load)
                total_price.append(env.total_power_cost)
            eval_data = pd.DataFrame()
            eval_data['Inside Temperatures 1'] = inside_temperatures_1
            eval_data['Inside Temperatures 2'] = inside_temperatures_2
            eval_data['Base Loads 1'] = base_loads_1
            eval_data['Base Loads 2'] = base_loads_2
            eval_data['Ambient Temperatures'] = ambient_temperatures
            eval_data['Rewards'] = rewards
            eval_data['Total Load'] = total_loads
            eval_data['Total Price'] = total_price
            with open(os.getcwd() + '/data/output/' + model_name + str(PRICE_SET[0]) +'_base_eval.pkl', 'wb') as f:
                pkl.dump(eval_data, f)

            #With constant price at 30€
            state = env.reset()
            inside_temperatures_1 = [state[2].item()]
            inside_temperatures_2 = [state[3].item()]
            base_loads_1 = [env.buildings[0].base_load]
            base_loads_2 = [env.buildings[1].base_load]
            ambient_temperatures = [env.ambient_temperature]
            total_loads = [env.total_load]
            total_price = [env.total_power_cost]
            rewards = [0]
            print('Starting evaluation of the model')
            price_index = 5
            for t_episode in range(NUM_TIME_STEPS):
                next_state, reward, done = env.step(price_index)
                rewards.append(reward)
                inside_temperatures_1.append(next_state[2])
                inside_temperatures_2.append(next_state[3])
                base_loads_1.append(env.buildings[0].base_load)
                base_loads_2.append(env.buildings[1].base_load)
                ambient_temperatures.append(env.ambient_temperature)
                total_loads.append(env.total_load)
                total_price.append(env.total_power_cost)
            eval_data = pd.DataFrame()
            eval_data['Inside Temperatures 1'] = inside_temperatures_1
            eval_data['Inside Temperatures 2'] = inside_temperatures_2
            eval_data['Base Loads 1'] = base_loads_1
            eval_data['Base Loads 2'] = base_loads_2
            eval_data['Ambient Temperatures'] = ambient_temperatures
            eval_data['Rewards'] = rewards
            eval_data['Total Load'] = total_loads
            eval_data['Total Price'] = total_price
            with open(os.getcwd() + '/data/output/' + model_name + str(PRICE_SET[price_index]) + 'base_eval.pkl', 'wb') as f:
                pkl.dump(eval_data, f)

            #With the spot prices
            state = env.reset()
            inside_temperatures_1 = [state[2].item()]
            inside_temperatures_2 = [state[3].item()]
            base_loads_1 = [env.buildings[0].base_load]
            base_loads_2 = [env.buildings[1].base_load]
            ambient_temperatures = [env.ambient_temperature]
            total_loads = [env.total_load]
            total_price = [env.total_power_cost]
            rewards = [0]
            print('Starting evaluation of the model')
            env.spot = True
            prices = pd.read_csv('data/environment/2014_ToU_prices.csv',
                                  header = 0).iloc[0:NUM_HOURS+1,1]
            for t_episode in range(NUM_TIME_STEPS):
                price = prices[t_episode]
                next_state, reward, done = env.step(price)
                rewards.append(reward)
                inside_temperatures_1.append(next_state[2])
                inside_temperatures_2.append(next_state[3])
                base_loads_1.append(env.buildings[0].base_load)
                base_loads_2.append(env.buildings[1].base_load)
                ambient_temperatures.append(env.ambient_temperature)
                total_loads.append(env.total_load)
                total_price.append(env.total_power_cost)

            eval_data = pd.DataFrame()
            eval_data['Inside Temperatures 1'] = inside_temperatures_1
            eval_data['Inside Temperatures 2'] = inside_temperatures_2
            eval_data['Base Loads 1'] = base_loads_1
            eval_data['Base Loads 2'] = base_loads_2
            eval_data['Ambient Temperatures'] = ambient_temperatures
            eval_data['Rewards'] = rewards
            eval_data['Total Load'] = total_loads
            eval_data['Total Price'] = total_price
            eval_data['Prices'] = prices
            with open(os.getcwd() + '/data/output/' + model_name +  '_ToU_base_eval.pkl', 'wb') as f:
                pkl.dump(eval_data, f)

            print('Finished evaluation on January.')

if __name__ == '__main__':
    args = parse_args()
    run(**vars(args))