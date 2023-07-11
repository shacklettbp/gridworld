import torch
import numpy as np

# Goal: define a policy class that samples actions in a way we can backprop/optimize or at least modify

# Policy class
class TabularPolicy(torch.nn.Module):
    def __init__(self, num_states, num_actions, greedy):
        super(TabularPolicy, self).__init__()
        self.num_actions = num_actions
        self.num_states = num_states
        self.greedy = greedy
        self.policy = torch.nn.Parameter(torch.ones(num_states, num_actions) / num_actions) # Start uniform

    def forward(self, states):
        # states is a batch of states
        # returns a batch of actions
        if self.greedy:
            return torch.argmax(self.policy[states], dim=1)
        else:
            #print(self.policy[states])
            return torch.multinomial(self.policy[states], 1)

    def update_policy(self, state, new_policy):
        self.policy[state] = new_policy
        return