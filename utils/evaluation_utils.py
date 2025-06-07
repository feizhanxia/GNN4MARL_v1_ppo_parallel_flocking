import torch
import time
import threading
from queue import Queue
from env.reward_env import FlockingEnv

class Evaluator:
    def __init__(self, policy, args):
        self.policy = policy
        self.args = args
        self.eval_device = torch.device("cpu")
        self.eval_env = self.create_eval_env()
        self.best_reward = float('-inf')
        
    def create_eval_env(self):
        """创建评估环境"""
        return FlockingEnv(
            n_agents=self.args.n_agents,
            box_size=self.args.box_size,
            radius=self.args.radius,
            dt=self.args.dt,
            speed=self.args.speed,
            physics_steps=self.args.physics_steps  # 新增
        )
    
    def evaluate_policy(self, n_episodes=10):
        """评估策略的性能，使用独立的CPU设备"""
        # 将策略复制到评估设备
        eval_policy = self.policy.to(self.eval_device)
        
        total_rewards = []
        for _ in range(n_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0
            for _ in range(self.args.steps_per_ep):
                x = torch.cat([obs['pos'], obs['vel']], dim=-1).to(self.eval_device)
                edge_index = obs['edge_index'].to(self.eval_device)
                
                with torch.no_grad():
                    action, _, _ = eval_policy.act(x, edge_index)
                obs, reward = self.eval_env.step(action.detach().cpu())
                episode_reward += reward.mean().item()
            
            total_rewards.append(episode_reward)
        
        return sum(total_rewards) / (len(total_rewards) * self.args.steps_per_ep)
    
    def run_evaluation(self, best_reward_queue, stop_event):
        """运行评估循环"""
        while not stop_event.is_set():
            time.sleep(self.args.eval_interval)
            if stop_event.is_set():
                break
                
            avg_reward = self.evaluate_policy(self.args.eval_episodes)
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                best_reward_queue.put((avg_reward, self.policy.state_dict()))
                print(f"🎉 新的最佳模型！平均奖励: {avg_reward:.4f}")
