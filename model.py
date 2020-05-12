import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class CentralizedCritic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(CentralizedCritic, self).__init__()

        # obs_dim = n_agents * local_obs_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.obs_dim, 1024)
        self.linear2 = nn.Linear(1024 + self.action_dim, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 1)

    def forward(self, x, a):

        x = F.relu(self.linear1(x))
        xa_cat = torch.cat([x,a], 1)
        xa = F.relu(self.linear2(xa_cat))
        xa = F.relu(self.linear3(xa))
        qval = self.linear4(xa)

        return qval

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, discrete):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.obs_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, self.action_dim)

        if discrete: # logits for discrete action
            self.out = lambda x: x
        else:
            # initialize small to prevent saturation
            self.linear3.weight.data.uniform_(-3e-3, 3e-3)
            self.out = F.sigmoid

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = self.out(self.linear3(x))

        return x