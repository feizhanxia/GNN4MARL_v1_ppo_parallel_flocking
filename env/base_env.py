# env/base_env.py

import numpy as np
import torch
from torch_geometric.nn import radius_graph

class SwarmBaseEnv:
    def __init__(self, n_agents=50, box_size=10.0, radius=1.0, dt=0.1, speed=0.05):
        self.n_agents = n_agents
        self.box_size = box_size
        self.radius = radius
        self.dt = dt
        self.speed = speed

        self.pos = None  # shape: [N, 2]
        self.vel = None  # shape: [N, 2], unit vectors

    def reset(self):
        g = torch.Generator()
        g.manual_seed(np.random.randint(0, 1e6))
        self.pos = torch.rand((self.n_agents, 2), generator=g) * self.box_size
        angles = torch.rand(self.n_agents, generator=g) * 2 * torch.pi
        self.vel = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        return self.get_observation()

    def step(self, angle_delta):
        """
        angle_delta: tensor [N], rotation (in radians) applied to each agent's current velocity
        """
        angle_delta = angle_delta.squeeze()
        current_angles = torch.atan2(self.vel[:, 1], self.vel[:, 0])
        new_angles = current_angles + angle_delta
        self.vel = torch.stack([torch.cos(new_angles), torch.sin(new_angles)], dim=1)

        # Update positions
        self.pos += self.vel * self.speed * self.dt
        self.pos %= self.box_size  # periodic boundary

        return self.get_observation()

    def get_observation(self):
        """
        Returns positions, velocities, and edge_index (radius graph).
        """
        edge_index = radius_graph(self.pos, r=self.radius, loop=True)
        return {
            'pos': self.pos.detach(),
            'vel': self.vel.detach(),
            'edge_index': edge_index.detach()
        }

    def get_state_tensor(self):
        """
        Concatenate pos and vel into a state vector for GNN input.
        """
        return torch.cat([self.pos, self.vel], dim=1)

    def to(self, device):
        self.pos = self.pos.to(device)
        self.vel = self.vel.to(device)
        return self
