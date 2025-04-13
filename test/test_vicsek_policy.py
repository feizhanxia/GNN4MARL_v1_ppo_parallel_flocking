# test/test_vicsek_policy.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from env.reward_env import FlockingEnv
from agents.vicsek_policy import VicsekPolicy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Setup
n_agents = 40
box_size = 10.0
radius = 1.5
dt = 0.2
speed = 1.0
steps = 300
env = FlockingEnv(n_agents=n_agents, speed=speed, box_size=box_size, dt=dt, radius=radius)
policy = VicsekPolicy(noise_scale=0.05)

obs = env.reset()
positions = [obs['pos'].clone()]

# --- Rollout
for _ in range(steps):
    actions = policy.act(obs)
    obs, reward = env.step(actions)
    positions.append(obs['pos'].clone())

# --- Visualization
fig, ax = plt.subplots(figsize=(6, 6))
scat = ax.scatter([], [], s=20)
ax.set_xlim(0, env.box_size)
ax.set_ylim(0, env.box_size)
ax.set_title("VicsekPolicy Flocking Test")

positions = torch.stack(positions)  # [T, N, 2]


def animate(i):
    ax.clear()
    ax.set_xlim(0, env.box_size)
    ax.set_ylim(0, env.box_size)
    ax.set_title(f"Step {i}")
    pos = positions[i]
    ax.scatter(pos[:, 0], pos[:, 1], s=20, color='green')

ani = animation.FuncAnimation(fig, animate, frames=len(positions), interval=50)
plt.show()
ani.save("vicsek_policy_test.mp4", writer='ffmpeg')  # Optional save