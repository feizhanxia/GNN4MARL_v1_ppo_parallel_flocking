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
from utils.visualization_utils import plot_flock_state

def parse_args():
    parser = argparse.ArgumentParser()
    # 测试参数
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--test_episodes", type=int, default=1)
    parser.add_argument("--steps_per_ep", type=int, default=300)
    parser.add_argument("--save_animation", action="store_true")
    parser.add_argument("--save_dir", type=str, default="test_results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()

def test_episode(env, policy, args, save_animation=False, save_dir=None):
    """运行一个测试episode"""
    obs = env.reset()
    positions = [obs['pos'].clone()]  # 记录位置历史
    velocities = [obs['vel'].clone()]  # 记录速度历史
    rewards = []
    
    for _ in range(args.steps_per_ep):
        x = torch.cat([obs['pos'], obs['vel']], dim=-1)
        edge_index = obs['edge_index']
        
        with torch.no_grad():
            action, _, _ = policy.act(x, edge_index)
        
        obs, reward = env.step(action)
        positions.append(obs['pos'].clone())
        velocities.append(obs['vel'].clone())
        rewards.append(reward.mean().item())
    
    # 转换为张量
    positions = torch.stack(positions)    # [T, N, 2]
    velocities = torch.stack(velocities)  # [T, N, 2]
    
    # 创建动画
    save_path = os.path.join(save_dir, f'flock_animation.mp4') if save_animation and save_dir else None
    create_animation(positions, velocities, env.box_size, save_path)
    
    return positions, rewards

def create_animation(positions, velocities, box_size, save_path=None):
    """创建群集运动的动画"""
    positions = positions.cpu().numpy()    # [T, N, 2]
    velocities = velocities.cpu().numpy()  # [T, N, 2]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        ax.clear()
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.grid(True)
        
        pos = positions[frame]      # [N, 2]
        vel = velocities[frame]     # [N, 2]
        
        # 画每个粒子的视野范围（浅蓝色圆圈）
        radius = 1.5  # 视野半径，与环境中的radius参数保持一致
        for i in range(len(pos)):
            circle = plt.Circle(pos[i], radius, color='lightblue', alpha=0.1)
            ax.add_patch(circle)
        
        # 画粒子（蓝色小圆点）
        ax.scatter(pos[:, 0], pos[:, 1], c='blue', s=30)  # 减小粒子大小
        
        # 画速度方向（红色箭头）
        # 归一化速度向量
        vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)
        vel_norm = np.where(vel_norm == 0, 1, vel_norm)
        vel_normalized = vel / vel_norm
        
        # 画箭头
        arrow_length = 0.3
        for i in range(len(pos)):
            ax.arrow(pos[i, 0], pos[i, 1],
                    vel_normalized[i, 0] * arrow_length,
                    vel_normalized[i, 1] * arrow_length,
                    head_width=0.1, head_length=0.15,
                    fc='red', ec='red', alpha=0.8)
        
        ax.set_title(f'Step {frame}')
    
    anim = FuncAnimation(fig, update, frames=len(positions),
                        interval=40, blit=False)
    
    # 显示动画
    plt.show()
    
    # 保存动画
    if save_path:
        anim.save(save_path, writer='ffmpeg')

def main():
    args = parse_args()
    
    # 加载模型和配置
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
    
    # 从检查点中获取模型配置
    model_config = ckpt.get('config', {})
    env_config = ckpt.get('config', {})
    
    if not model_config:
        print("\n警告: 未找到模型配置，使用默认值")
        model_config = {
            'in_dim': 4,
            'hidden_dim': 8,  # 改为8以匹配训练时的设置
            'std': 0.1
        }
    
    if not env_config:
        print("\n警告: 未找到环境配置，使用默认值")
        env_config = {
            'n_agents': 10,
            'box_size': 5.0,
            'radius': 3.0,
            'dt': 0.05,
            'speed': 1.0
        }
    
    # 创建环境
    env = FlockingEnv(
        n_agents=env_config.get('n_agents', 40),
        box_size=env_config.get('box_size', 10.0),
        radius=env_config.get('radius', 1.5),
        dt=env_config.get('dt', 0.05),
        speed=env_config.get('speed', 1.0)
    )
    
    # 创建策略网络
    policy = ParallelGNNPolicyAC(
        in_dim=model_config.get('in_dim', 4),
        hidden_dim=model_config.get('hidden_dim', 64),
        std=model_config.get('std', 0.1),
        device=device
    )
    
    # 加载模型权重
    try:
        policy.load_state_dict(model_state_dict)
        print(f"\n成功加载模型: {args.ckpt_path}")
    except Exception as e:
        print(f"\n加载模型权重失败: {e}")
        return
    
    print(f"最佳奖励: {ckpt.get('best_reward', 'N/A')}")
    print("\n环境配置:")
    for k, v in env_config.items():
        print(f"  {k}: {v}")
    print("\n模型配置:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    # 运行测试
    all_rewards = []
    for ep in range(args.test_episodes):
        print(f"\n测试 Episode {ep + 1}/{args.test_episodes}")
        positions, rewards = test_episode(
            env, policy, args,
            save_animation=args.save_animation,  # 直接使用命令行参数
            save_dir=args.save_dir
        )
        all_rewards.append(rewards)
        
        # 打印统计信息
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"平均奖励: {mean_reward:.4f} ± {std_reward:.4f}")
    
    # 绘制奖励曲线
    if args.save_dir and all_rewards:
        os.makedirs(args.save_dir, exist_ok=True)
        plt.figure(figsize=(10, 5))
        for ep, rewards in enumerate(all_rewards):
            plt.plot(rewards, alpha=0.5, label=f'Episode {ep+1}')
        plt.plot(np.mean(all_rewards, axis=0), 'k-', label='Mean')
        plt.fill_between(
            range(len(all_rewards[0])),
            np.mean(all_rewards, axis=0) - np.std(all_rewards, axis=0),
            np.mean(all_rewards, axis=0) + np.std(all_rewards, axis=0),
            alpha=0.2
        )
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'test_rewards.png'))
        plt.close()

if __name__ == "__main__":
    main() 