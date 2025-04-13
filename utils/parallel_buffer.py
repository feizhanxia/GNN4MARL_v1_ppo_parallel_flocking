import torch
import numpy as np

class ParallelBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def store(self, state, action, log_prob, reward, value, done):
        """存储单个时间步的数据"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def get_tensors(self, device):
        """获取所有数据并转换为张量"""
        return {
            'states': torch.stack(self.states).to(device),
            'actions': torch.stack(self.actions).to(device),
            'log_probs': torch.stack(self.log_probs).to(device),
            'rewards': torch.stack(self.rewards).to(device),
            'values': torch.stack(self.values).to(device),
            'dones': torch.stack(self.dones).to(device)
        }
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = [] 