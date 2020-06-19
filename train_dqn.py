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

def train_dqn(ckpt,model_name,dynamic,soft, RL):
    env = System(RL_building=RL)
    scores = []
    brain = DAgent(gamma=GAMMA, epsilon=EPSILON, batch_size=BATCH_SIZE, n_actions=N_ACTIONS,
                  input_dims=INPUT_DIMS,  lr = LEARNING_RATE, eps_dec = EPS_DECAY, ckpt=ckpt)
    for i_episode in range(NUM_EPISODES):
        # Initialize the environment.rst and state
        state = env.reset()
        state = torch.tensor(state,dtype=torch.float).to(device)
        # Normalizing data using an online algo
        brain.normalizer.observe(state)
        state = brain.normalizer.normalize(state).unsqueeze(0)
        score = 0
        for t in count():
            # Select and perform an action
            action = brain.select_action(state).type(torch.FloatTensor)
            next_state, reward, done = env.step(action.item())
            score += reward
            reward = torch.tensor([reward],dtype=torch.float,device=device)

            if not done:
                next_state = torch.tensor(next_state,dtype=torch.float, device=device)
                #normalize data using an online algo
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

        #sys.stdout.write('Finished episode {} with reward {}\n'.format(i_episode, score))
        with open(os.getcwd() + '/data/output/' + model_name + '_rewards.txt', 'a') as f:
            f.write('Episode {}, Reward {} \n'.format(i_episode, score))
        # Soft update for target network:

        if soft:
            brain.soft_update(TAU)

        # Update the target network, copying all weights and biases in DQN
        else:
            if i_episode % TARGET_UPDATE == 0:
                 brain.target_net.load_state_dict(brain.policy_net.state_dict())


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
        pkl.dump(scores,f)


    # Saving the final model
    torch.save(brain, os.getcwd() + model_name + 'model.pt')
    print('Complete')

    brain.epsilon = 0
    brain.eps_end = 0

    inside_temperatures_1 = []
    inside_temperatures_2 = []
    ambient_temperatures = []
    times = []
    base_loads = []
    actions = []

    for inside_temp_1 in np.arange(18, 22, 1 / 5):
        print(inside_temp_1)
        for inside_temp_2 in np.arange(18, 22, 1 / 5):
            for ambient_temp in np.arange(-5, 5, 1 / 5):
                for load in np.arange(0, 30, 1/2):
                    for time in range(0,24):
                        scaled_load = load/1000
                        state = [ambient_temp, scaled_load, inside_temp_1, inside_temp_2, time]
                        state = torch.tensor(state, dtype=torch.float).to(device)
                        state = brain.normalizer.normalize(state).unsqueeze(0)
                        action = brain.select_action(state).type(torch.FloatTensor).item()
                        inside_temperatures_1.append(inside_temp_1)
                        inside_temperatures_2.append(inside_temp_2)
                        ambient_temperatures.append(ambient_temp)
                        times.append(time)
                        base_loads.append(load)
                        actions.append(action)

    eval_data = pd.DataFrame()
    eval_data['Inside Temperatures 1'] = inside_temperatures_1
    eval_data['Inside Temperatures 2'] = inside_temperatures_2
    eval_data['Ambient Temperatures'] = ambient_temperatures
    eval_data['Times'] = times
    eval_data['Loads'] = base_loads
    eval_data['Actions'] = actions
    with open(os.getcwd() + '/data/output/' + model_name + 'policy_eval.pkl', 'wb') as f:
        pkl.dump(eval_data, f)

    env = System(eval=True, january=january, RL_building=RL)

    # base_loads_2 = [env.buildings[1].base_load]
    actions_building_1 = []
    actions_building_2 = []
    ambient_temperatures = [env.ambient_temperature]
    total_loads = []
    total_price = []
    actions = []
    rewards = []
    print('Starting evaluation of the model')
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float).to(device)
    inside_temperatures_1 = [state[2].item()]
    inside_temperatures_2 = [state[3].item()]
    base_loads = [state[1].item()]
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
        base_loads.append(next_state[1])
        # base_loads_2.append(env.buildings[1].base_load)
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
    # actions.append(0)
    # total_loads.append(env.total_load)
    eval_data = pd.DataFrame()
    eval_data['Inside Temperatures 1'] = inside_temperatures_1[:-1]
    eval_data['Inside Temperatures 2'] = inside_temperatures_2[:-1]
    eval_data['Base Loads'] = base_loads[:-1]
    # eval_data['Base Loads 2'] = base_loads_2
    eval_data['Actions 1'] = actions_building_1
    eval_data['Actions 2'] = actions_building_2
    eval_data['Ambient Temperatures'] = ambient_temperatures[:-1]
    eval_data['Actions'] = actions
    eval_data['Rewards'] = rewards
    eval_data['Total Load'] = total_loads
    eval_data['Total Price'] = total_price
    with open(os.getcwd() + '/data/output/' + model_name + '_eval.pkl', 'wb') as f:
        pkl.dump(eval_data, f)

    # Evaluation if price was kept constant
    # env = System(eval=True)
    state = env.reset()
    actions_building_1 = []
    actions_building_2 = []
    inside_temperatures_1 = [state[2].item()]
    inside_temperatures_2 = [state[3].item()]
    base_loads = [state[1].item()]
    # base_loads_2 = [env.buildings[1].base_load]
    ambient_temperatures = [env.ambient_temperature]
    total_loads = []
    total_price = []
    rewards = []
    print('Starting evaluation of the model')
    for t_episode in range(NUM_TIME_STEPS):
        next_state, reward, done = env.step(0)
        rewards.append(reward)
        inside_temperatures_1.append(next_state[2])
        inside_temperatures_2.append(next_state[3])
        actions_building_1.append(env.buildings[0].action)
        actions_building_2.append(env.buildings[1].action)
        base_loads.append(next_state[1])
        # base_loads_2.append(env.buildings[1].base_load)
        ambient_temperatures.append(env.ambient_temperature)
        total_loads.append(env.total_load)
        total_price.append(env.total_power_cost)
    # total_loads.append(env.total_load)
    eval_data = pd.DataFrame()
    eval_data['Inside Temperatures 1'] = inside_temperatures_1[:-1]
    eval_data['Inside Temperatures 2'] = inside_temperatures_2[:-1]
    eval_data['Base Loads'] = base_loads[:-1]
    # eval_data['Base Loads 2'] = base_loads_2
    eval_data['Actions 1'] = actions_building_1
    eval_data['Actions 2'] = actions_building_2
    eval_data['Ambient Temperatures'] = ambient_temperatures[:-1]
    eval_data['Rewards'] = rewards
    eval_data['Total Load'] = total_loads
    eval_data['Total Price'] = total_price
    with open(os.getcwd() + '/data/output/' + model_name + str(PRICE_SET[0]) + '_base_eval.pkl', 'wb') as f:
        pkl.dump(eval_data, f)

    # With constant price at 30€
    state = env.reset()
    actions_building_1 = []
    actions_building_2 = []
    inside_temperatures_1 = [state[2].item()]
    inside_temperatures_2 = [state[3].item()]
    base_loads = [state[1].item()]
    # base_loads_2 = [env.buildings[1].base_load]
    ambient_temperatures = [env.ambient_temperature]
    total_loads = []
    total_price = []
    rewards = []
    print('Starting evaluation of the model')
    price_index = 5
    for t_episode in range(NUM_TIME_STEPS):
        next_state, reward, done = env.step(price_index)
        rewards.append(reward)
        inside_temperatures_1.append(next_state[2])
        inside_temperatures_2.append(next_state[3])
        actions_building_1.append(env.buildings[0].action)
        actions_building_2.append(env.buildings[1].action)
        base_loads.append(next_state[1])
        # base_loads_2.append(env.buildings[1].base_load)
        ambient_temperatures.append(env.ambient_temperature)
        total_loads.append(env.total_load)
        total_price.append(env.total_power_cost)
    eval_data = pd.DataFrame()
    eval_data['Inside Temperatures 1'] = inside_temperatures_1[:-1]
    eval_data['Inside Temperatures 2'] = inside_temperatures_2[:-1]
    eval_data['Base Loads'] = base_loads[:-1]
    # eval_data['Base Loads 2'] = base_loads_2
    eval_data['Actions 1'] = actions_building_1
    eval_data['Actions 2'] = actions_building_2
    eval_data['Ambient Temperatures'] = ambient_temperatures[:-1]
    eval_data['Rewards'] = rewards
    eval_data['Total Load'] = total_loads
    eval_data['Total Price'] = total_price
    with open(os.getcwd() + '/data/output/' + model_name + str(PRICE_SET[price_index]) + 'base_eval.pkl', 'wb') as f:
        pkl.dump(eval_data, f)

    # With the spot prices
    state = env.reset()
    actions_building_1 = []
    actions_building_2 = []
    inside_temperatures_1 = [state[2].item()]
    inside_temperatures_2 = [state[3].item()]
    base_loads = [state[1].item()]
    # base_loads_2 = [env.buildings[1].base_load]
    ambient_temperatures = [env.ambient_temperature]
    total_loads = []
    total_price = []
    rewards = []
    print('Starting evaluation of the model')
    env.spot = True
    prices = pd.read_csv('data/environment/2014_ToU_prices.csv',
                         header=0).iloc[0:NUM_HOURS + 1, 1]
    for t_episode in range(NUM_TIME_STEPS):
        price = prices[t_episode]
        next_state, reward, done = env.step(price)
        rewards.append(reward)
        inside_temperatures_1.append(next_state[2])
        inside_temperatures_2.append(next_state[3])
        actions_building_1.append(env.buildings[0].action)
        actions_building_2.append(env.buildings[1].action)
        base_loads.append(next_state[1])
        # base_loads_2.append(env.buildings[1].base_load)
        ambient_temperatures.append(env.ambient_temperature)
        total_loads.append(env.total_load)
        total_price.append(env.total_power_cost)

    eval_data = pd.DataFrame()
    eval_data['Inside Temperatures 1'] = inside_temperatures_1[:-1]
    eval_data['Inside Temperatures 2'] = inside_temperatures_2[:-1]
    eval_data['Base Loads'] = base_loads[:-1]
    eval_data['Actions 1'] = actions_building_1
    eval_data['Actions 2'] = actions_building_2
    # eval_data['Base Loads 2'] = base_loads_2
    eval_data['Ambient Temperatures'] = ambient_temperatures[:-1]
    eval_data['Rewards'] = rewards
    eval_data['Total Load'] = total_loads
    eval_data['Total Price'] = total_price
    eval_data['Prices'] = prices
    with open(os.getcwd() + '/data/output/' + model_name + '_ToU_base_eval.pkl', 'wb') as f:
        pkl.dump(eval_data, f)

    print('Finished evaluation on January.')