# multi-building-RL

This repo implements a deep reinforcement learning based price-setting mechanism for electricity helping to perform demand-response. The agent sends out prices for electricity to houses that have differable loads. These loads can be controlled using simple rules or by a reinforcement learning agent trained using [the single building repository](https://github.com/Cernewein/heating-RL-agent). It features both a [Deep Q-Learning](https://arxiv.org/abs/1312.5602) algorithm as well as a [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) solution. Built with Python & Pytorch.


It has been developped for my master thesis at the Technical University of Denmark.
A paper based on the work done during the master thesis has been published [here](https://doi.org/10.1145/3427773.3427862).

# Installation

In order to install the needed packages run:
```
pip install -r requirement.txt
```


# Environment & Problem formulation

![](/images/aggregationPb.PNG)

Below is the objective function of the overall optimization problem. The costs for the individual buildings are shown in green, while the cost for the aggregator is shown in red. The total system cost is the cost of both part each weighted using $\zeta$ . $g_{t}$ represents the load that trespasses the aggregator's capacity, and $\lambda_L$ is a weighting factor. See [the single building repository](https://github.com/Cernewein/heating-RL-agent) for more details on the cost for the individual buildings.

![](/images/objectiveFunction.PNG)



# RL formulation

The price setting agent has only one possible action : it can choose the electricity price level. The prices are either continuous in the case of the DDPG agent, or discrete in the case of the DQN agent.
![](/images/RLFormulation.PNG)


# Data

The dataset used for modelling the environment contains three distinct parts (chosen year is 2014 but can of course be any):
* The historic electricity spot prices (region DK1 and DK2) for the year 2014 have been obtained on [NordPool](https://www.nordpoolgroup.com)
* The historic electricity loads (region DK1 and DK2) for the year 2014 have been obtained on [NordPool](https://www.nordpoolgroup.com)
* The historic weather conditions (temperature and sun) for Copenhagen in the year 2014 have been obtained on [RenawablesNinja](https://www.renewables.ninja/)


# Results

The prices that are set by the aggregation node are shown in the top row of the figure below. The base load as well as the shiftable load under different pricing mechanisms is shown on the second row. Inside and outside temperatures are shown in third and forth row.

The aggregation agent learns to set prices according the total load and is able to shift the heating load to lower base load periods. The overall price level is also correlated to the outside temperature, as lower temperatures mean higher heating demand. The inside temperatures for both houses remain at a reasonable level.Lower comfort bounds are : 
* House 1 -> 20 °C
* House 2 -> 19 °C

![](/images/twoBuildingsLoad.png)
