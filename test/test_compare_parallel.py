import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.reward_env import FlockingEnv
from agents.parallel_policy_ac import ParallelGNNPolicyAC
from agents.random_policy import RandomPolicy
from agents.vicsek_policy import VicsekPolicy
from utils.visualization_utils import plot_flock_state

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save_animation", action="store_true")
    parser.add_argument("--save_dir", type=str, default="test_results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def create_animation_single(positions, velocities, box_size, radius, policy_name, save_path=None):
    positions = positions.cpu().numpy()
    velocities = velocities.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    color = {'Random': 'gray', 'Vicsek': 'green', 'GNN': 'blue'}[policy_name]

    def update(frame):
        ax.clear()
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.grid(False)

        pos = positions[frame]
        vel = velocities[frame]

        for i in range(len(pos)):
            circle = plt.Circle(pos[i], radius, color=color, alpha=0.05)
            ax.add_patch(circle)

        ax.scatter(pos[:, 0], pos[:, 1], c=color, s=30, label=policy_name)

        vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)
        vel_norm = np.where(vel_norm == 0, 1, vel_norm)
        vel_normalized = vel / vel_norm

        for i in range(len(pos)):
            ax.arrow(pos[i, 0], pos[i, 1],
                     vel_normalized[i, 0] * 0.6,
                     vel_normalized[i, 1] * 0.6,
                     head_width=0.1, head_length=0.15,
                     fc=color, ec=color, alpha=0.8)

        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=True)
        ax.set_title(f'{policy_name} - Step {frame}')

    anim = FuncAnimation(fig, update, frames=len(positions), interval=35, blit=False)

    if save_path:
        anim.save(save_path, writer='ffmpeg')
    else:
        plt.show()

def test_policy(env, policy, steps, policy_type='GNN'):
    """测试单个策略"""
    obs = env.reset()
    positions = [obs['pos'].clone()]
    velocities = [obs['vel'].clone()]
    rewards = []
    
    for _ in range(steps):
        if policy_type == 'GNN':
            x = torch.cat([obs['pos'], obs['vel']], dim=-1)
            edge_index = obs['edge_index']
            with torch.no_grad():
                action, _, _ = policy.act(x, edge_index)
        else:
            action = policy.act(obs)
        
        obs, reward = env.step(action)
        positions.append(obs['pos'].clone())
        velocities.append(obs['vel'].clone())
        rewards.append(reward.mean().item())
    
    return torch.stack(positions), torch.stack(velocities), rewards

def main(env_args):
    args = parse_args()
    
    # 加载GNN模型
    device = torch.device(args.device)
    try:
        ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"加载模型文件失败: {e}")
        return
    
    # 检查检查点内容
    print("\n检查点内容的键:")
    for key in ckpt.keys():
        print(f"  - {key}")
    
    # 尝试不同的可能的键名
    model_state_dict = None
    for key in ['model', 'state_dict', 'model_state_dict', 'network', 'policy']:
        if key in ckpt:
            model_state_dict = ckpt[key]
            print(f"\n使用键 '{key}' 加载模型状态")
            break
    
    if model_state_dict is None:
        print("\n错误: 在检查点中找不到模型状态字典")
        if len(ckpt.keys()) == 1:
            # 如果只有一个键，可能直接就是模型状态
            model_state_dict = ckpt
            print("尝试直接使用检查点作为模型状态")
    
    # 获取配置
    model_config = ckpt.get('config', {})

    # 创建三个相同的环境
    envs = {
        'Random': FlockingEnv(**env_args),
        'Vicsek': FlockingEnv(**env_args),
        'GNN': FlockingEnv(**env_args)
    }
    
    # 创建三种策略
    gnn_policy = ParallelGNNPolicyAC(
        in_dim=model_config.get('in_dim', 4),
        hidden_dim=model_config.get('hidden_dim', 8),
        std=model_config.get('std', 0.1),
        device=device
    )
    
    # 加载模型权重
    try:
        gnn_policy.load_state_dict(model_state_dict)
        print(f"\n成功加载模型: {args.ckpt_path}")
    except Exception as e:
        print(f"\n加载模型权重失败: {e}")
        return
    
    policies = {
        'Random': RandomPolicy(n_agents=env_args['n_agents']),
        'Vicsek': VicsekPolicy(noise_scale=0.05),
        'GNN': gnn_policy
    }
    
    print(f"最佳奖励: {ckpt.get('best_reward', 'N/A')}")
    print("\n环境配置:")
    for k, v in env_args.items():
        print(f"  {k}: {v}")
    print("\n模型配置:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    # 测试所有策略
    all_positions = {}
    all_velocities = {}
    all_rewards = {}
    

    for name, policy in policies.items():
        print(f"\n测试 {name} 策略...")
        positions, velocities, rewards = test_policy(
            envs[name], policy, args.steps,
            policy_type='GNN' if name == 'GNN' else 'Other'
        )
        all_positions[name] = positions
        all_velocities[name] = velocities
        all_rewards[name] = rewards
        print(f"{name} 平均奖励: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")

        # 为每个策略分别创建动画
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            anim_path = os.path.join(args.save_dir, f'{name.lower()}_animation.mp4')
        else:
            anim_path = None

        create_animation_single(positions, velocities,
                                env_args['box_size'], env_args['radius'],
                                policy_name=name, save_path=anim_path)

    
    # 绘制奖励对比曲线
    plt.figure(figsize=(10, 5), dpi=300)
    for name, rewards in all_rewards.items():
        plt.plot(rewards, label=name, 
                color={'Random': 'gray', 'Vicsek': 'green', 'GNN': 'blue'}[name],
                linestyle='--' if name == 'Random' else '-')
    
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Policy Comparison')
    plt.grid(True)
    plt.legend()
    
    if args.save_dir:
        plt.savefig(os.path.join(args.save_dir, 'compare_rewards.png'))
    plt.show()

if __name__ == "__main__":
    # 创建环境和策略
    env_args = {
        'n_agents': 40,
        'box_size': 10.0,
        'radius': 1.5,
        'dt': 0.05,
        'speed': 1.0
    }
    main(env_args) 