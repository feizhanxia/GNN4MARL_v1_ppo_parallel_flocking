import torch
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def setup_training_directory(args):
    """设置训练目录和日志"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"
    save_dir = os.path.join(args.save_root, run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(save_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("version: handcrafted_single_PPO\n")
    
    # 设置TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))
    print(f"✅ TensorBoard logs will be saved to: {os.path.join(save_dir, 'tb')}")
    print("👉 Run this anytime: tensorboard --logdir", args.save_root, "\n")
    
    return save_dir, writer

def save_model(save_dir, policy, config, filename, best_reward=None):
    """保存模型"""
    save_dict = {
        'model_state_dict': policy.state_dict(),
        'config': config
    }
    if best_reward is not None:
        save_dict['best_reward'] = best_reward
    
    torch.save(save_dict, os.path.join(save_dir, filename))

def update_learning_rate(optimizer, current_episode, total_episodes, initial_lr, min_lr):
    """线性衰减学习率"""
    new_lr = max(min_lr, initial_lr * (1 - (current_episode + 1) / total_episodes))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr 