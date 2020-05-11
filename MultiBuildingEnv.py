import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time
from vars import *
import random

class env:
    def __init__(self, eval = False, num_buildings = 2):
        self.observation_space = np.array([5,5,5])# Observation space size of aggregator, building 1 and building 2
        self.action_space = np.array([1,1,1]) # Number of actions one agent can do
        self.n_agents = 3
        self.eval = eval
        # If we are in eval mode, select the month of january
        if self.eval:
            self.random_day = 0  # First day of the year
        else:
            # Else select November/December for training
            self.random_day = random.randint(304, 365 - NUM_HOURS // 24 - 1) * 24
        self.ambient_temperatures = pd.read_csv(
            '../heating-RL-agent/data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
            header=3).iloc[self.random_day:self.random_day + NUM_HOURS + 1, 2]
        self.ambient_temperature = self.ambient_temperatures[self.random_day]
        ### Based on the same day, choose the sun irradiation for the episode

        self.sun_powers = pd.read_csv(
            '../heating-RL-agent/data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
            header=3).iloc[self.random_day:self.random_day + NUM_HOURS + 1, 3]

        self.buildings = [Building(random_day=self.random_day, ambient_temperatures=self.ambient_temperatures,
                                   sun_powers=self.sun_powers, name='Building_{}'.format(b), seed=b) for b in
                          range(num_buildings)]

        self.done = False
        self.time = 0
        self.total_load = sum(self.buildings[i].base_load for i in range(num_buildings))
        self.zeta = ZETA

    def step(self, actions):
        # Actions are received as an array, [aggregator action, building action 1, building action 2]
        chosen_price = actions[0].detach().item()*MAX_PRICE + 10 # Minimum price is 10, can go up to 10 + MAX_PRICE

        building_costs = []
        building_loads = []
        building_new_states = []
        building_dones = []

        for b,building in enumerate(self.buildings):
            new_state, cost, load, done = building.step(actions[b].detach().item(), chosen_price)
            building_costs.append(cost)
            building_loads.append(load)
            building_new_states.append(new_state)
            building_dones.append(done)

        total_load = sum(building_loads)

        aggregator_reward = self.reward(total_load, building_costs)

        self.time += 1
        if self.time >= NUM_TIME_STEPS:
            self.done = True

        aggregator_new_state = [self.ambient_temperature, total_load, self.buildings[0].inside_temperature,
                                 self.buildings[1].inside_temperature,
                                 self.time % int(24 * 3600 // TIME_STEP_SIZE)]

        all_new_states = [aggregator_new_state, building_new_states[0], building_new_states[1]]

        return all_new_states, [aggregator_reward,building_costs[0], building_costs[1]], [self.done] +  building_dones


    def reward(self, total_load, building_costs):
        penalty = np.maximum(0, total_load - L_MAX)
        penalty *= LOAD_PENALTY

        return  self.zeta * sum(building_costs) - (1-self.zeta) * penalty

    def reset(self):
        building_states = []
        total_load = 0
        for b, building in enumerate(self.buildings):
            new_state, load = building.reset(self.random_day, self.ambient_temperatures, self.sun_powers)
            building_states.append(new_state)
            total_load += load

        if self.eval:
            self.random_day = 0 # First day of the year
        else:
            # Else select November/December for training
            self.random_day=random.randint(304,365-NUM_HOURS//24-1)*24
        self.ambient_temperatures = pd.read_csv(
            '../heating-RL-agent/data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
            header=3).iloc[self.random_day:self.random_day+NUM_HOURS+1,2]
        self.ambient_temperature = self.ambient_temperatures[self.random_day]
        ### Based on the same day, choose the sun irradiation for the episode

        self.sun_powers = pd.read_csv(
            '../heating-RL-agent/data/environment/ninja_weather_55.6838_12.5354_uncorrected.csv',
            header=3).iloc[self.random_day:self.random_day+NUM_HOURS+1,3]

        self.done = False
        self.time = 0
        total_load = 0

        return [[self.ambient_temperature, total_load, self.buildings[0].inside_temperature, self.buildings[1].inside_temperature,
                self.time % int(24 * 3600 // TIME_STEP_SIZE)]] + building_states

class Building:
    """ This class represents the building that has to be controlled. Its dynamics are modelled based on an RC analogy.
    When instanciated, it initialises the inside temperature to 21°C, the envelope temperature to 20, and resets the done
    and time variables.
    """
    def __init__(self, random_day, ambient_temperatures, sun_powers, name='', seed = 0):

        # If variable sun power, outside temperature, price should be used
        self.random_day = random_day
        self.name = name

        ### Initiliazing the temperatures
        self.inside_temperature = 21.0 #np.random.randint(19,24)

        ### Selecting a random set for the outside temperatures based on a dataset

        self.ambient_temperatures = ambient_temperatures
        self.ambient_temperature = self.ambient_temperatures[random_day]
        self.sun_powers = sun_powers
        self.sun_power = self.sun_powers[random_day]

        self.base_loads = pd.read_csv(
            '../multi-building-RL/data/environment/2014_DK2_scaled_loads.csv',header=0).iloc[random_day:random_day+NUM_HOURS+1,1]
        self.seed = seed
        np.random.seed(self.seed)
        self.T_MIN = T_MIN - seed*0.5
        self.base_loads += np.random.normal(loc=0.0, scale=0.075/1000, size=NUM_HOURS+1)
        self.base_load = self.base_loads[random_day]

        ## How much power from the grid has been drawn?
        self.power_from_grid = 0
        self.done = False
        self.time = 0


    def heat_pump_power(self, phi_e):
        """Takes an electrical power flow and converts it to a heat flow.

        :param phi_e: The electrical power
        :type phi_e: Float
        :return: Returns the heat flow as an integer
        """
        return phi_e*(0.0606*self.ambient_temperature+2.612)

    def step(self, action, price):
        """

        :param action: The chosen action - is the index of selected action from the action space.
        :type action: Integer
        :return: Returns the new state after a step, the reward for the action and the done state
        """

        delta = 1 / (R_IA * C_I) * (self.ambient_temperature - self.inside_temperature) + \
                self.heat_pump_power(NOMINAL_HEAT_PUMP_POWER*action)/C_I + A_w*self.sun_power/C_I

        self.inside_temperature += delta * TIME_STEP_SIZE

        # After having updated storage, battery power is scaled to MW for price computation

        # Heat pump power is adjusted so that the power is expressed in MW and also adjusted to the correct time slot size
        heat_pump_power = action * NOMINAL_HEAT_PUMP_POWER / (1e6) * TIME_STEP_SIZE / 3600

        # Power drawn from grid is the sum of what is put into battery or drawn from battery and how much we are heating
        # It is also minus what has been produced by PV if PV mode is activated

        total_load = (heat_pump_power + self.base_load * TIME_STEP_SIZE / 3600)

        self.power_from_grid = heat_pump_power

        r = self.reward(self.power_from_grid, price)

        # Updating the outside temperature with the new temperature
        self.ambient_temperature = self.ambient_temperatures[self.random_day + (self.time * TIME_STEP_SIZE)//3600]
        self.sun_power = self.sun_powers[self.random_day + (self.time * TIME_STEP_SIZE)//3600]
        self.base_load = self.base_loads[self.random_day + (self.time * TIME_STEP_SIZE) // 3600]
        self.time += 1
        if self.time >= NUM_TIME_STEPS:
            self.done = True

        return [self.inside_temperature, self.ambient_temperature, self.sun_power, self.base_load, (self.time * TIME_STEP_SIZE) // 3600], r, total_load, self.done #


    def reward(self,power_from_grid, price):
        """
        Returns the received value for the chosen action and transition to next state

        :param action: The selected action
        :return: Returns the reward for that action
        """

        penalty = np.maximum(0,self.inside_temperature-T_MAX) + np.maximum(0,T_MIN-self.inside_temperature)
        penalty *= COMFORT_PENALTY

        paid_price = - power_from_grid*price

        reward =  paid_price - penalty

        return reward

    def reset(self, random_day, ambient_temperatures, sun_powers):
        """
        This method is resetting the attributes of the building.

        :return: Returns the resetted inside temperature, ambient temperature and sun power
        """
        self.inside_temperature = 21

        self.random_day = random_day
        self.ambient_temperatures = ambient_temperatures
        self.ambient_temperature = self.ambient_temperatures[random_day]
        self.sun_powers = sun_powers
        self.sun_power = self.sun_powers[random_day]

        self.base_loads = pd.read_csv(
            '../multi-building-RL/data/environment/2014_DK2_scaled_loads.csv',
            header=0).iloc[random_day:random_day+NUM_HOURS+1,1]

        self.base_loads += np.random.normal(loc=0.0, scale=0.075/1000, size=NUM_HOURS+1)
        self.base_load = self.base_loads[random_day]
        self.time = 0

        return [self.inside_temperature, self.ambient_temperature, self.sun_power, self.base_load, self.time], self.base_load