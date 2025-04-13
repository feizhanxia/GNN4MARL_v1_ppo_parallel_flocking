# agents/random_policy.py

import torch
from agents.policy_base import BasePolicy

class RandomPolicy(BasePolicy):
    def __init__(self, n_agents, angle_limit=3.1415):
        self.n_agents = n_agents
        self.angle_limit = angle_limit  # maximum angle delta

    def act(self, obs):
        """
        Returns a random rotation angle delta in range [-angle_limit, angle_limit]
        """
        return (torch.rand(self.n_agents) * 2 - 1) * self.angle_limit

    def update(self):
        pass  # No learning for random policy

    def store_transition(self, *args, **kwargs):
        pass  # No memory for random policy
