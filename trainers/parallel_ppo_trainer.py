import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.parallel_buffer import ParallelBuffer

class ParallelPPOTrainer:
    def __init__(self, policy, gamma=0.99, lam=0.95, lr=1e-3, radius=1.0,
                 clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, epochs=5):
        self.policy = policy
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.epochs = epochs
        self.radius = radius
        
        self.optimizer = Adam(policy.parameters(), lr=lr)
        self.buffer = ParallelBuffer()
    
    def compute_gae(self, rewards, values, dones):
        """计算广义优势估计"""
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def update_from_batch(self, batch):
        """从批量数据更新策略"""
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        rewards = batch['rewards']
        values = batch['values']
        dones = batch['dones']
        edge_index_list = batch['edge_index']  # 每个时间步的edge_index列表
        
        # 计算GAE
        advantages = self.compute_gae(rewards, values, dones)  # [T, N]
        returns = advantages + values  # [T, N]
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # [T, N]
        advantages = advantages.unsqueeze(-1)  # [T, N, 1]
        
        # 计算新的动作概率和价值
        new_log_probs = []
        entropies = []
        new_values = []
        
        # 对每个时间步分别计算
        for i in range(len(states)):
            log_prob, entropy, value = self.policy.evaluate_actions(states[i], edge_index_list[i], actions[i])
            new_log_probs.append(log_prob)
            entropies.append(entropy)
            new_values.append(value)
        
        # 堆叠结果
        new_log_probs = torch.stack(new_log_probs)  # [T, N, 1]
        entropy = torch.stack(entropies).mean()  # 标量
        new_values = torch.stack(new_values)  # [T, N, 1]
        
        # 计算比率
        ratio = torch.exp(new_log_probs - old_log_probs)  # [T, N, 1]
        
        # 计算PPO目标
        surr1 = ratio * advantages  # [T, N, 1]
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages  # [T, N, 1]
        policy_loss = -torch.min(surr1, surr2).mean()  # 标量
        
        # 计算价值损失
        value_loss = F.mse_loss(new_values, returns.unsqueeze(-1))  # 标量
        
        # 计算总损失
        total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy  # 标量
        
        # 更新策略
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.item(), total_loss.item() 