import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed, hidden_sizes = [64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (list): List of number of nodes in each hidden layer
        """
        super(Actor, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.layers = nn.ModuleList([nn.Linear(self.state_size, hidden_sizes[0])]) # first layer
        self.layers.extend([nn.Linear(s1, s2) for s1, s2 in 
                            zip(hidden_sizes[:-1], hidden_sizes[1:])])             # middle layers
        self.out = nn.Linear(hidden_sizes[-1], self.action_size)                   # last layer
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights."""
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.out.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        return F.tanh(self.out(x))
    
    
class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed, hidden_sizes = [64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (list): List of number of nodes in each hidden layer
        """
        super(Critic, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.layers = nn.ModuleList([nn.Linear(self.state_size, hidden_sizes[0])]) # first layer
        h_sizes = hidden_sizes.copy()                                              # create copy of list!
        h_sizes[0] += action_size                                                  # input of 2. layer
        self.layers.extend([nn.Linear(s1, s2) for s1, s2 in 
                            zip(h_sizes[:-1], h_sizes[1:])])                       # middle layers
        self.out = nn.Linear(h_sizes[-1], self.action_size)                        # final layer
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights."""
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.out.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """Build a network that maps state -> action values."""
        x = F.relu(self.layers[0](state))                                          # first layer
        x = torch.cat((x,action), dim=1)                                           # concat with actions   
        for layer in self.layers[1:]:                                              # middle layers
            x = F.relu(layer(x))
        return F.tanh(self.out(x))                                                 # final layer
