import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time
from vars import *
import random


class System:
    def __init__(self, eval = False, num_buildings = 2, zeta = 0.5):
        self.eval = eval
        # If we are in eval mode, select the month of january
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


        self.buildings = [Building(random_day = self.random_day,ambient_temperatures = self.ambient_temperatures,
                                   sun_powers = self.sun_powers, name = 'Building_{}'.format(b)) for b in range(num_buildings)]

        self.zeta = zeta
        self.done = False
        self.time = 0

    def step(self, action):

        price = PRICE_SET[int(action)]

        total_load, building_costs = self.get_loads_and_costs(price)

        self.ambient_temperature = self.ambient_temperatures[self.random_day + (self.time * TIME_STEP_SIZE) // 3600]

        r = self.reward(total_load, building_costs)
        self.time +=1

        if self.time >= NUM_TIME_STEPS:
            self.done = True


        return [self.ambient_temperature, total_load,
                self.time % int(24 * 3600 // TIME_STEP_SIZE)], r, self.done  #

    def get_loads_and_costs(self, action):
        total_load = 0
        total_cost = 0
        for building in self.buildings:
            load, cost = building.step(action)
            total_load += load
            total_cost += cost

        return total_load, total_cost

    def reward(self, total_load, building_costs):
        penalty = np.maximum(0, total_load - L_MAX)
        penalty *= LOAD_PENALTY
        print(penalty)

        return  - self.zeta * building_costs - (1-self.zeta) * penalty

    def reset(self):
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
        for building in self.buildings:
            total_load += building.reset(self.random_day, self.ambient_temperatures, self.sun_powers)

        return [self.ambient_temperature, total_load,
                self.time % int(24 * 3600 // TIME_STEP_SIZE)]



class Building:
    """ This class represents the building that has to be controlled. Its dynamics are modelled based on an RC analogy.
    When instanciated, it initialises the inside temperature to 21°C, the envelope temperature to 20, and resets the done
    and time variables.
    """
    def __init__(self,random_day, ambient_temperatures, sun_powers,  name = ''):
        self.random_day = random_day
        self.ambient_temperatures = ambient_temperatures
        self.ambient_temperature = self.ambient_temperatures[random_day]
        self.sun_powers = sun_powers
        self.sun_power = self.sun_powers[random_day]
        self.name = name
        self.time = 0
        self.base_loads = pd.read_csv(
            '../multi-building-RL/data/environment/2014_DK2_scaled_loads.csv',header=0).iloc[random_day:random_day+NUM_HOURS+1,1]
        self.base_loads += np.random.normal(loc=0.0, scale=0.075/1000, size=NUM_HOURS+1)
        self.base_load = self.base_loads[random_day]
        self.inside_temperature = 21


    def heat_pump_power(self, phi_e):
        """Takes an electrical power flow and converts it to a heat flow.

        :param phi_e: The electrical power
        :type phi_e: Float
        :return: Returns the heat flow as an integer
        """
        return phi_e*(0.0606*self.ambient_temperature+2.612)

    def step(self, price):
        """

        :param action: The chosen action - is the index of selected action from the action space.
        :type action: Integer
        :return: Returns the new state after a step, the reward for the action and the done state
        """

        current_penalty = COMFORT_PENALTY * (np.maximum(0,T_MIN-self.inside_temperature))
        expected_cost = (PRICE_SENSITIVITY * NOMINAL_HEAT_PUMP_POWER / (1e6) * price * TIME_STEP_SIZE / 3600)

        if current_penalty -  expected_cost> 0:
            selected_action = 1
        else:
            selected_action = 0


        delta = 1 / (R_IA * C_I) * (self.ambient_temperature - self.inside_temperature) + \
                self.heat_pump_power(NOMINAL_HEAT_PUMP_POWER*selected_action)/C_I + A_w*self.sun_power/C_I

        self.inside_temperature += delta * TIME_STEP_SIZE

        # Heat pump power is adjusted so that the power is expressed in MW and also adjusted to the correct time slot size
        heat_pump_power = selected_action * NOMINAL_HEAT_PUMP_POWER / (1e6) * TIME_STEP_SIZE / 3600
        total_load = (heat_pump_power + self.base_load)
        total_cost = total_load*price + current_penalty

        self.time +=1

        self.ambient_temperature = self.ambient_temperatures[self.random_day + (self.time * TIME_STEP_SIZE)//3600]
        self.sun_power = self.sun_powers[self.random_day + (self.time * TIME_STEP_SIZE)//3600]
        self.base_load = self.base_loads[self.random_day + (self.time * TIME_STEP_SIZE) // 3600]
        return total_load, total_cost

    def reset(self, random_day ,ambient_temperatures, sun_powers):
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
        self.time = 0
        self.base_loads = pd.read_csv(
            '../multi-building-RL/data/environment/2014_DK2_scaled_loads.csv',
            header=0).iloc[random_day:random_day+NUM_HOURS+1,1]
        self.base_loads += np.random.normal(loc=0.0, scale=0.075/1000, size=NUM_HOURS+1)
        self.base_load = self.base_loads[random_day]
        return self.base_load



