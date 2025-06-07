import torch
import argparse
import threading
from queue import Queue
import torch.multiprocessing as mp
import numpy as np
import signal
import sys
import time

from env.reward_env import FlockingEnv
from agents.parallel_policy_ac import ParallelGNNPolicyAC
from trainers.parallel_ppo_trainer import ParallelPPOTrainer
from utils.seed import set_seed
from utils.training_utils import setup_training_directory, save_model, update_learning_rate
from utils.evaluation_utils import Evaluator
# from utils.visualization_utils import plot_reward_curve

# 全局变量用于存储奖励历史
# reward_history = []

def signal_handler(sig, frame):
    """处理中断信号"""
    print("\n正在停止训练...")
    sys.exit(0)

def worker_func(worker_id, args, policy_state_dict, sample_queue, stop_event):
    """工作进程函数"""
    try:
        # 创建本地环境
        env = FlockingEnv(
            n_agents=args.n_agents,
            box_size=args.box_size,
            radius=args.radius,
            dt=args.dt,
            speed=args.speed,
            physics_steps=args.physics_steps  # 新增
        )
        
        # 创建本地策略网络
        device = torch.device("cpu")  # 工作进程只使用CPU
        policy = ParallelGNNPolicyAC(
            in_dim=4,
            hidden_dim=args.hidden_dim,
            std=args.std,
            device=device
        )
        
        while not stop_event.is_set():
            # 从共享内存加载策略状态
            policy.load_state_dict(dict(policy_state_dict))
            
            # 执行一个episode的采样
            obs = env.reset()
            states, actions, log_probs, rewards, values, dones, edge_indices = [], [], [], [], [], [], []
            
            for _ in range(args.steps_per_ep):
                x = torch.cat([obs['pos'], obs['vel']], dim=-1)
                edge_index = obs['edge_index']
                
                with torch.no_grad():
                    action, log_prob, value = policy.act(x, edge_index)
                
                obs, reward = env.step(action)
                
                # 收集数据
                states.append(x)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value.squeeze(-1))
                dones.append(torch.zeros_like(reward))
                edge_indices.append(edge_index)
            
            # 将数据放入队列
            episode_data = {
                'states': torch.stack(states),
                'actions': torch.stack(actions),
                'log_probs': torch.stack(log_probs),
                'rewards': torch.stack(rewards),
                'values': torch.stack(values),
                'dones': torch.stack(dones),
                'edge_index': edge_indices
            }
            sample_queue.put(episode_data)
            
            # 清理内存
            del states, actions, log_probs, rewards, values, dones, edge_indices
            del episode_data
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
    except Exception as e:
        print(f"工作进程 {worker_id} 出错: {e}")
        raise

class ParallelSampler:
    def __init__(self, policy, args, n_workers=4):
        self.policy = policy
        self.args = args
        self.n_workers = n_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        
        # 创建进程间通信队列，限制大小为工作进程数的2倍
        self.sample_queue = mp.Queue(maxsize=n_workers * 2)
        self.stop_event = mp.Event()
        
        # 创建共享内存来存储策略状态
        self.policy_state_dict = mp.Manager().dict()
        self.update_policy_state_dict(policy.state_dict())
        
        # 创建进程池
        self.processes = []
        for i in range(n_workers):
            p = mp.Process(
                target=worker_func,
                args=(i, args, self.policy_state_dict, self.sample_queue, self.stop_event)
            )
            p.daemon = True
            self.processes.append(p)
    
    def update_policy_state_dict(self, new_state_dict):
        """更新共享内存中的策略状态"""
        for k, v in new_state_dict.items():
            self.policy_state_dict[k] = v
    
    def update_policy(self, new_policy_state_dict):
        """更新工作进程的策略网络"""
        self.update_policy_state_dict(new_policy_state_dict)
    
    def start(self):
        """启动所有工作进程"""
        for p in self.processes:
            p.start()
    
    def stop(self):
        """停止所有工作进程"""
        self.stop_event.set()
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive():
                    p.kill()
    
    def collect_samples(self, n_samples):
        """收集指定数量的样本"""
        samples = []
        while len(samples) < n_samples:
            try:
                # 设置较短的超时时间，避免队列积压
                sample = self.sample_queue.get(timeout=0.1)
                samples.append(sample)
            except:
                if self.stop_event.is_set():
                    break
                continue
        
        if not samples:
            raise RuntimeError("无法收集到足够的样本")
        
        # 合并所有样本
        batch = {
            'states': torch.cat([s['states'] for s in samples]),
            'actions': torch.cat([s['actions'] for s in samples]),
            'log_probs': torch.cat([s['log_probs'] for s in samples]),
            'rewards': torch.cat([s['rewards'] for s in samples]),
            'values': torch.cat([s['values'] for s in samples]),
            'dones': torch.cat([s['dones'] for s in samples]),
            'edge_index': [edge_index for s in samples for edge_index in s['edge_index']]
        }
        
        # 清理内存
        del samples
        return batch

def run_parallel_ppo_training(args):
    # 设置中断信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 设置随机种子和设备
    set_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    # 创建模型
    in_dim = 4
    policy = ParallelGNNPolicyAC(in_dim=in_dim, hidden_dim=args.hidden_dim, std=args.std, device=device)
    
    trainer = ParallelPPOTrainer(
        policy,
        gamma=args.gamma,
        lam=args.lam,
        lr=args.lr,
        radius=args.radius,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        epochs=args.epochs,
    )
    
    # 设置训练目录和日志
    save_dir, writer = setup_training_directory(args)
    
    # 初始化并行采样器
    sampler = ParallelSampler(policy, args, n_workers=args.n_workers)
    sampler.start()
    
    # 初始化评估器
    evaluator = Evaluator(policy, args)
    best_reward_queue = Queue()
    stop_event = threading.Event()
    
    # 启动评估线程
    eval_thread = threading.Thread(
        target=evaluator.run_evaluation,
        args=(best_reward_queue, stop_event)
    )
    eval_thread.daemon = True
    eval_thread.start()
    
    try:
        for ep in range(args.episodes):
            # 收集并行样本
            batch = sampler.collect_samples(args.n_workers)
            
            # 将数据移到训练设备
            batch = {
                'states': batch['states'].to(device),
                'actions': batch['actions'].to(device),
                'log_probs': batch['log_probs'].to(device),
                'rewards': batch['rewards'].to(device),
                'values': batch['values'].to(device),
                'dones': batch['dones'].to(device),
                'edge_index': batch['edge_index']  # edge_index保持原样，因为它是列表
            }
            
            # 更新策略
            policy_loss, value_loss, entropy_loss, total_loss = trainer.update_from_batch(batch)
            
            # 更新共享内存中的策略
            sampler.update_policy(policy.state_dict())
            
            # 更新学习率
            new_lr = update_learning_rate(trainer.optimizer, ep, args.episodes, args.lr, args.min_lr)
            
            # 计算平均奖励
            avg_reward = batch['rewards'].mean().item()
            avg_action = batch['actions'].mean().item()
            std_action = batch['actions'].std().item()
            
            # 记录训练信息
            # reward_history.append(avg_reward)
            writer.add_scalar("Loss/Total", total_loss, ep)
            writer.add_scalar("Loss/Policy", policy_loss, ep)
            writer.add_scalar("Loss/Value", value_loss, ep)
            writer.add_scalar("Loss/Entropy", entropy_loss, ep)
            writer.add_scalar("Average Reward", avg_reward, ep)
            writer.add_scalar("Action/Average", avg_action, ep)
            writer.add_scalar("Action/Std", std_action, ep)
            writer.add_scalar("Train/Learning Rate", new_lr, ep)
            writer.add_scalar("Policy/log_std", policy.log_std.item(), ep)
            
            # 保存最佳模型
            while not best_reward_queue.empty():
                best_reward, best_state_dict = best_reward_queue.get()
                save_model(
                    save_dir,
                    policy,
                    {'in_dim': in_dim, 'hidden_dim': args.hidden_dim, 'std': args.std},
                    "policy_best.pt",
                    best_reward
                )
                print(f"💾 保存最佳模型，奖励: {best_reward:.4f}")
            
            # 打印训练信息
            print(f"Ep {ep+1}/{args.episodes} | Reward: {avg_reward:.4f} | Loss: {total_loss:.4f}")
            
            # 定期保存检查点
            if (ep + 1) % args.ckpt_interval == 0:
                save_model(
                    save_dir,
                    policy,
                    {'in_dim': in_dim, 'hidden_dim': args.hidden_dim, 'std': args.std},
                    f"policy_ep{ep+1}.pt"
                )
    
    except Exception as e:
        print(f"训练出错: {e}")
        raise
    finally:
        stop_event.set()
        eval_thread.join()
        sampler.stop()
    
    # 保存最终模型和绘制奖励曲线
    save_model(
        save_dir,
        policy,
        {'in_dim': in_dim, 'hidden_dim': args.hidden_dim, 'std': args.std},
        "policy_final.pt"
    )
    # plot_reward_curve(reward_history, save_dir, args.episodes)
    writer.close()
    print("\n✅ Training complete. Final model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=50)
    parser.add_argument("--box_size", type=float, default=10.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--physics_steps", type=int, default=1)  # 新增
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--std", type=float, default=0.1)
    parser.add_argument("--min_std", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps_per_ep", type=int, default=300)
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save_root", type=str, default="training_logs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--eval_interval", type=int, default=60, help="评估间隔（秒）")
    parser.add_argument("--eval_episodes", type=int, default=10, help="评估次数")
    parser.add_argument("--n_workers", type=int, default=4, help="并行采样工作进程数")
    args = parser.parse_args()
    
    run_parallel_ppo_training(args)