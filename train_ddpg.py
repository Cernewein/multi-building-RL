from DDPG import DDPGagent
import sys
from environment import *
from matplotlib import style
import numpy as np
style.use('ggplot')
from vars import *
from itertools import count
import pickle as pkl
import os
import argparse
import sys
import torch
import pandas as pd


def train_ddpg(model_name, RL = True):
    env = System(RL_building=RL, continuous = True)
    scores = []
    brain = DDPGagent(mem_size=MEMORY_SIZE, add_noise = False)

    for i_episode in range(NUM_EPISODES):
        # Initialize the environment.rst and state
        #brain.reset()
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float).to(device)
        # Normalizing data using an online algo
        brain.normalizer.observe(state)
        state = brain.normalizer.normalize(state).unsqueeze(0)
        score = 0
        for t in count():
            # Select and perform an action
            action = brain.select_action(state).type(torch.FloatTensor)
            next_state, reward, done = env.step(action.item())
            score += reward
            reward = torch.tensor([reward], dtype=torch.float, device=device)
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                # normalize data using an online algo
                brain.normalizer.observe(next_state)
                next_state = brain.normalizer.normalize(next_state).unsqueeze(0)

            else:
                next_state = None

            # Store the transition in memory
            brain.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            brain.optimize_model()
            if done:
                scores.append(score)
                break

        with open(os.getcwd() + '/data/output/' + model_name + '_rewards.txt', 'a') as f:
            f.write('Episode {}, Reward {} \n'.format(i_episode, score))

        if i_episode == 0:
            best_score = score
        else:
            if score > best_score:
                # Save current best model
                best_score = score
                torch.save(brain, os.getcwd() + model_name + 'model.pt')

        #sys.stdout.write('Finished episode {} with reward {}\n'.format(i_episode, score))


    model_params = {'NUM_EPISODES':NUM_EPISODES,
                    'EPSILON':EPSILON,
                    'EPS_DECAY':EPS_DECAY,
                    'LEARNING_RATE_':LEARNING_RATE,
                    'GAMMA':GAMMA,
                    'TARGET_UPDATE':TARGET_UPDATE,
                    'BATCH_SIZE':BATCH_SIZE,
                     'TIME_STEP_SIZE':TIME_STEP_SIZE,
                    'NUM_HOURS':NUM_HOURS,
                    'COMFORT_PENALTY':COMFORT_PENALTY,
                    'LOAD_PENALTY':LOAD_PENALTY,
                    'PRICE_PENALTY':PRICE_PENALTY,
                    'ZETA':ZETA}


    scores.append(model_params)
    with open(os.getcwd() + '/data/output/' + model_name + '_dynamic_' + str(dynamic) + '_rewards_dqn.pkl', 'wb') as f:
        pkl.dump(scores, f)


    # Saving the final model
    torch.save(brain, os.getcwd() + model_name + '_final_model.pt')
    print('Complete')

    brain.epsilon = 0
    brain.eps_end = 0
    env = System(eval=True, january=True, RL_building=RL, continuous=True)

    inside_temperatures_1 = [env.buildings[0].inside_temperature]
    inside_temperatures_2 = [env.buildings[NUM_BUILDINGS//2].inside_temperature]
    base_loads_1 = [env.buildings[0].base_load]
    base_loads_2 = [env.buildings[NUM_BUILDINGS//2].base_load]
    actions_building_1 = [0]
    actions_building_2 = [0]
    ambient_temperatures = [env.ambient_temperature]
    total_loads = [env.total_load]
    actions = [0]
    rewards = [0]
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
        inside_temperatures_2.append(env.buildings[NUM_BUILDINGS//2].inside_temperature)
        actions_building_1.append(env.buildings[0].action)
        actions_building_2.append(env.buildings[NUM_BUILDINGS//2].action)
        base_loads_1.append(env.buildings[0].base_load)
        base_loads_2.append(env.buildings[NUM_BUILDINGS//2].base_load)
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
    eval_data['Actions 1'] = actions_building_1
    eval_data['Actions 2'] = actions_building_2
    eval_data['Ambient Temperatures'] = ambient_temperatures
    eval_data['Actions'] = actions
    eval_data['Rewards'] = rewards
    eval_data['Total Load'] = total_loads
    with open(os.getcwd() + '/data/output/' + model_name + '_eval.pkl', 'wb') as f:
        pkl.dump(eval_data, f)

    # Evaluation if price was kept constant
    # env = System(eval=True)
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
    with open(os.getcwd() + '/data/output/' + model_name + '_base_eval.pkl', 'wb') as f:
        pkl.dump(eval_data, f)

    print('Finished evaluation on January.')