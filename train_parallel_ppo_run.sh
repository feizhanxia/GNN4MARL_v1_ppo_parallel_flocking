#!/bin/zsh

# 🧠 Environment config
N_AGENTS=40
BOX_SIZE=10.0
RADIUS=1.5
DT=0.2
SPEED=1.0
PHYSICS_STEPS=5  # 新增：每个动作之间的物理更新步数

# 🧠 Model config
HIDDEN_DIM=8
STD=0.5
MIN_STD=$STD

# 🎯 PPO hyperparameters
GAMMA=0.995
LAM=0.995
LR=5e-3
MIN_LR=1e-4
CLIP_EPS=0.15
VF_COEF=0.001
ENT_COEF=1e-3
EPOCHS=1

# 🧪 Training config
EPISODES=100
STEPS_PER_EP=250
CKPT_INTERVAL=200
SEED=36

# 📊 Evaluation config
EVAL_INTERVAL=100
EVAL_EPISODES=10

# 🔄 Parallel config
N_WORKERS=10  # 并行采样工作进程数

# 💻 System
DEVICE=cpu  # or 'cpu'
SAVE_ROOT=training_logs_parallel  # 使用不同的保存目录

python train_parallel_ppo.py \
  --n_agents $N_AGENTS \
  --box_size $BOX_SIZE \
  --radius $RADIUS \
  --dt $DT \
  --speed $SPEED \
  --physics_steps $PHYSICS_STEPS \
  --hidden_dim $HIDDEN_DIM \
  --std $STD \
  --min_std $MIN_STD \
  --gamma $GAMMA \
  --lam $LAM \
  --lr $LR \
  --min_lr $MIN_LR \
  --clip_eps $CLIP_EPS \
  --vf_coef $VF_COEF \
  --ent_coef $ENT_COEF \
  --epochs $EPOCHS \
  --episodes $EPISODES \
  --steps_per_ep $STEPS_PER_EP \
  --ckpt_interval $CKPT_INTERVAL \
  --device $DEVICE \
  --save_root $SAVE_ROOT \
  --seed $SEED \
  --eval_interval $EVAL_INTERVAL \
  --eval_episodes $EVAL_EPISODES \
  --n_workers $N_WORKERS