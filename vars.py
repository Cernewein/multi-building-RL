import torch
### General settings
TIME_STEP_SIZE = 60*60# How many seconds are in one of our timeteps? For example if we want every minute, set this to 60
NUM_HOURS = 31*24
NUM_TIME_STEPS = int(NUM_HOURS*3600//TIME_STEP_SIZE) # A total of 12 hours computed every second

##### RL Agent parameters
NUM_EPISODES = 1000 # Number of episodes
EPSILON = 1 # For epsilon-greedy approach
EPS_DECAY = 0.99997
LEARNING_RATE = 0.00025
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.99
TARGET_UPDATE = 10
BATCH_SIZE = 32
PRICE_SET = [10,20,30,40,50,60]
N_ACTIONS = len(PRICE_SET)
INPUT_DIMS = 5
FC_1_DIMS = 100
FC_2_DIMS = 200
FC_3_DIMS = FC_2_DIMS # If we don't want a third layer, set this to FC_2_DIMS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TAU = 0.001 # For soft update
MEMORY_SIZE = 1000*31*24
ZETA = 0.2


##### Environment parameters
COMFORT_PENALTY = 5 # Penalty applied when going outside of "comfort" bounds
T_MIN = 19.5 # Minimum temperature that should be achieved inside of the building
T_MAX = 22.5 # Maximum temperature that should be achieved inside of the building
C_I = 2.07*3.6e6 # Based on Emil Larsen's paper - heat capacity of the building
C_E = 3.24*3.6e6 # Based on Emil Larsen's paper - heat capacity of the building
R_IA = 5.29e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
R_IE = 0.909e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
R_EA = 4.47e-3 # Thermal resistance between interior and ambient. Based on Emil Larsen's paper
A_w = 7.89 # Window surface area
NOMINAL_HEAT_PUMP_POWER = 2000 # 2kW based on some quick loockup of purchaseable heat pumps
PRICE_SENSITIVITY = 200
L_MAX = 5 / 1000 # N kWh scales down to MWh
LOAD_PENALTY = 1000
HEATING_SETTINGS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]