import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

class ParallelGNNPolicyAC(nn.Module):
    def __init__(self, in_dim, hidden_dim, std=0.1, device='cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # 策略网络
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, 1)
        
        # 价值网络
        self.value_conv1 = GCNConv(in_dim, hidden_dim)
        self.value_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # 动作标准差
        self.log_std = nn.Parameter(torch.ones(1) * np.log(std))
        
        # 初始化参数
        self._init_weights()
        self.to(device)
    
    def _init_weights(self):
        """初始化网络参数"""
        # 初始化GCN层
        for conv in [self.conv1, self.conv2, self.value_conv1, self.value_conv2]:
            nn.init.orthogonal_(conv.lin.weight, gain=np.sqrt(2))
            if conv.lin.bias is not None:
                nn.init.constant_(conv.lin.bias, 0)
        
        # 初始化线性层
        for linear in [self.policy_head, self.value_head]:
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            if linear.bias is not None:
                nn.init.constant_(linear.bias, 0)
    
    def forward(self, x, edge_index):
        # 策略网络前向传播
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        action_mean = self.policy_head(h)
        
        # 价值网络前向传播
        v = F.relu(self.value_conv1(x, edge_index))
        v = F.relu(self.value_conv2(v, edge_index))
        value = self.value_head(v)
        
        return action_mean, value
    
    def act(self, x, edge_index):
        """采样动作并计算对数概率"""
        action_mean, value = self.forward(x, edge_index)
        
        # 使用重参数化技巧采样动作
        std = torch.exp(self.log_std)
        noise = torch.randn_like(action_mean)
        action = action_mean + noise * std
        
        # 计算对数概率
        log_prob = self._log_prob(action, action_mean, std)
        
        # 将动作限制在[-pi, pi]范围内
        action = torch.clamp(action, -np.pi, np.pi)
        
        return action, log_prob, value
    
    def _log_prob(self, action, mean, std):
        """计算动作的对数概率"""
        var = std.pow(2)
        log_scale = torch.log(std)
        return -((action - mean).pow(2) / (2 * var) + log_scale + np.log(np.sqrt(2 * np.pi)))
    
    def evaluate_actions(self, states, edge_index, actions):
        """评估给定状态和动作的对数概率、熵和价值"""
        # 前向传播
        action_mean, value = self.forward(states, edge_index)
        std = torch.exp(self.log_std)
        
        # 计算对数概率
        log_prob = self._log_prob(actions, action_mean, std)
        
        # 计算熵
        entropy = 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std
        
        return log_prob, entropy, value
    
    def get_value(self, x, edge_index):
        """仅获取状态价值"""
        _, value = self.forward(x, edge_index)
        return value 