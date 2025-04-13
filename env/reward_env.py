# env/reward_env.py

from .base_env import SwarmBaseEnv
import torch
import torch.nn.functional as F


class FlockingEnv(SwarmBaseEnv):
    def __init__(self, n_agents=50, box_size=10.0, radius=1.0, dt=0.1, speed=0.05):
        super().__init__(n_agents, box_size, radius, dt, speed)
        self.last_angle_delta = None

    def step(self, angle_delta):
        obs = super().step(angle_delta)
        reward = self._compute_reward(obs, angle_delta)
        return obs, reward

    def _compute_reward(self, obs, angle_delta):
        """计算奖励
        
        奖励由四部分组成：
        1. 局部速度对齐 (40%): R_align ∈ [-1, 1]
        2. 局部聚集 (30%): R_cohesion ∈ [0, 1]
        3. 全局序参量 (20%): R_global ∈ [-1, 1]
        4. 平滑控制 (10%): R_smooth ∈ [-0.1π, 0]
        
        总奖励范围：[-0.91, 0.9]
        """
        pos = obs['pos']  # [N, 2]
        vel = obs['vel']  # [N, 2]
        edge_index = obs['edge_index']  # [2, E]
        N = pos.size(0)
        
        row, col = edge_index  # [E], [E]
        
        # --- 1. 局部速度对齐 ---
        vel_i = vel[row]  # [E, 2]
        vel_j = vel[col]  # [E, 2]
        align = F.cosine_similarity(vel_i, vel_j, dim=1)  # [E]
        align_score = torch.zeros(N, device=pos.device).scatter_add(0, row, align)  # [N]
        neighbor_count = torch.bincount(row, minlength=N).clamp(min=1)  # [N]
        align_score = align_score / neighbor_count  # [N]
        
        # --- 2. 局部聚集 ---
        pos_i = pos[row]  # [E, 2]
        pos_j = pos[col]  # [E, 2]
        dist = torch.norm(pos_i - pos_j, dim=1)  # [E]
        optimal_dist = self.radius * 0.1
        sigma = self.radius * 0.2
        cohesion = torch.exp(-(dist - optimal_dist)**2 / (2 * sigma**2))  # [E]
        cohesion_score = torch.zeros(N, device=pos.device).scatter_add(0, row, cohesion)  # [N]
        cohesion_score = cohesion_score / neighbor_count  # [N]
        
        # --- 3. 全局序参量 ---
        mean_vel = vel.mean(dim=0)  # [2]
        mean_vel = F.normalize(mean_vel, dim=0)  # [2]
        global_align = F.cosine_similarity(vel, mean_vel.unsqueeze(0), dim=1)  # [N]
        
        # --- 4. 平滑控制 ---
        angle_smoothness = -0.1 * torch.abs(angle_delta).squeeze(-1)  # [N]
        
        # --- trivial reward for test: fixed direction ---
        # target = -0.3
        # fixed_angle = torch.ones_like(angle_delta) * target  # 固定方向
        # angle_error = angle_delta - fixed_angle
        # angle_penalty = 40-20*torch.abs(angle_error).squeeze(-1)  # [N]
        
        # 组合奖励
        reward = (0.0 * align_score +      # 局部对齐
                 0.4 * cohesion_score +    # 局部聚集
                 0.6 * global_align +      # 全局对齐
                 0.1 * angle_smoothness)  # 平滑控制
        
        return reward
