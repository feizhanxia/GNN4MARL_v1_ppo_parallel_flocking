# agents/vicsek_policy.py

import torch
from agents.policy_base import BasePolicy

class VicsekPolicy(BasePolicy):
    def __init__(self, noise_scale=0.0):
        self.noise_scale = noise_scale  # Gaussian noise on angle

    def act(self, obs):
        pos, vel, edge_index = obs['pos'], obs['vel'], obs['edge_index']
        row, col = edge_index
        N = pos.size(0)

        # average neighbor directions
        vel_j = vel[col]
        sum_dir = torch.zeros_like(vel)
        sum_dir = sum_dir.index_add(0, row, vel_j)

        norm = sum_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        new_dir = sum_dir / norm

        # compute angle difference between current and new direction
        current_angle = torch.atan2(vel[:, 1], vel[:, 0])
        target_angle = torch.atan2(new_dir[:, 1], new_dir[:, 0])
        delta = target_angle - current_angle

        # wrap to [-pi, pi]
        delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi

        # add noise
        delta += torch.randn_like(delta) * self.noise_scale
        return delta.squeeze()

    def store_transition(self, *args, **kwargs):
        pass

    def update(self):
        pass
