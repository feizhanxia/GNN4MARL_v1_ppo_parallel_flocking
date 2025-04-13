import matplotlib.pyplot as plt
import os

def plot_reward_curve(reward_history, save_path, episodes):
    """绘制奖励曲线"""
    plt.figure(figsize=(10, 5), dpi=300)
    plt.scatter(range(episodes), reward_history, s=1, color='blue', label='Episode Reward')
    
    if len(reward_history) > 20:
        smoothed = [sum(reward_history[i-20:i])/20 for i in range(20, len(reward_history))]
        plt.plot(range(20, episodes), smoothed, color='red', label='SMA (20)')
    
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("PPO Training Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "reward_curve.png"))
    plt.close()

def plot_flock_state(pos, vel, edge_index, ax):
    """绘制群集状态
    
    Args:
        pos: 智能体位置 [N, 2]
        vel: 智能体速度 [N, 2]
        edge_index: 邻接矩阵 [2, E]
        ax: matplotlib轴对象
    """
    # 绘制智能体之间的连接
    for i in range(edge_index.shape[1]):
        start = pos[edge_index[0, i]]
        end = pos[edge_index[1, i]]
        ax.plot([start[0], end[0]], [start[1], end[1]], 'gray', alpha=0.2)
    
    # 绘制智能体位置和速度
    ax.scatter(pos[:, 0], pos[:, 1], c='b', s=50)
    ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1],
              color='r', scale=20, width=0.003)
    
    # 设置图像范围和样式
    ax.set_aspect('equal')
    ax.grid(True)
    return ax 