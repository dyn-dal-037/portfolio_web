# LEAP Hand Dexterous Grasping via Deep RL
### RRC IIIT-H | Technical Evaluation Task | Dhyan Dalwadi | IIIT Bhopal ECE '27

---

## Overview

This project implements a learning-based dexterous grasping pipeline for the **LEAP Hand** (16-DOF) using **Proximal Policy Optimization (PPO)** in **MuJoCo**. The system is trained to establish stable multi-finger contact with a target object and lift it to a specified height. A ROS 2 deployment pipeline (Task B) bridges the trained policy to real-time joint visualization in RViz2.

**Stack:**
- Simulator: MuJoCo (CPU-compatible, hardware-agnostic)
- RL Library: Stable Baselines3 — PPO
- Training: Google Colab (T4 GPU)
- Deployment: ROS 2 Humble | Ubuntu 22.04
- Hand Model: LEAP Hand URDF from `mujoco_menagerie`

---

## Repository Structure

```
leap_grasp_rl/
│
├── env/
│   ├── leap_grasp_env.py        # Custom Gymnasium environment
│   ├── reward.py                # Modular reward components
│   └── utils.py                 # Contact detection, pose helpers
│
├── train/
│   ├── train_ppo.ipynb          # Colab training notebook
│   ├── train_ppo.py             # Local training script
│   └── hyperparams.yaml         # PPO hyperparameter config
│
├── eval/
│   ├── evaluate.py              # Policy rollout + metrics
│   └── record_video.py          # Offscreen rendering for submission video
│
├── ros2_ws/
│   └── src/
│       └── leap_hand_deploy/
│           ├── leap_hand_deploy/
│           │   ├── policy_node.py       # Inference node → /hand/joint_commands
│           │   └── interface_node.py    # JointState bridge → RViz2
│           ├── launch/
│           │   └── leap_visualize.launch.py
│           ├── urdf/
│           │   └── leap_hand.urdf
│           └── config/
│               └── rviz_config.rviz
│
├── assets/
│   ├── learning_curve.png       # [PLACEHOLDER — add after training]
│   ├── grasp_demo.gif           # [PLACEHOLDER — add after eval]
│   └── rviz_demo.gif            # [PLACEHOLDER — add after ROS2 demo]
│
├── report/
│   └── report.pdf               # [PLACEHOLDER — final submission report]
│
├── requirements.txt
└── README.md
```

---

## Task A: RL-Based Dexterous Grasping

### Environment Design

#### Observation Space (42-dim continuous)

| Component | Dim | Description |
|---|---|---|
| Joint positions | 16 | All LEAP Hand DOF (rad) |
| Joint velocities | 16 | All LEAP Hand DOF (rad/s) |
| Object position | 3 | XYZ in world frame |
| Object orientation | 4 | Quaternion |
| Palm-to-object vector | 3 | Relative displacement |

**Total: 42-dimensional continuous observation vector**

#### Action Space (16-dim continuous)

Continuous PD position targets for all 16 hand joints, clipped to `[-1, 1]` and mapped to joint limits. PD controller runs at 200Hz simulation frequency; policy queries at 20Hz (10 sim steps per policy step).

#### Reward Function

```
r(t) = w₁ · r_approach + w₂ · r_contact + w₃ · r_lift + w₄ · r_smooth
```

| Component | Formula | Weight | Purpose |
|---|---|---|---|
| Approach | `-‖p_palm - p_obj‖` | 0.30 | Draw palm toward object |
| Contact | `Σ finger_i ∈ contact` | 0.30 | Encourage multi-finger contact |
| Lift | `max(0, z_obj - z_table)` | 0.50 | Primary task objective |
| Smoothness | `-0.01 · ‖a_t‖²` | 1.00 | Penalize joint torque waste |

**Episode termination:** object drops below table, 500 steps elapsed, or lift height > 0.15m (success).

---

### PPO Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `learning_rate` | 3e-4 | Standard Adam LR for continuous control |
| `n_steps` | 2048 | Rollout buffer per env per update |
| `batch_size` | 64 | Minibatch size for gradient updates |
| `n_epochs` | 10 | PPO update epochs per rollout |
| `gamma` | 0.99 | Discount — long horizon task |
| `gae_lambda` | 0.95 | GAE bias-variance tradeoff |
| `clip_range` | 0.2 | PPO clipping parameter |
| `ent_coef` | 0.01 | Entropy bonus for exploration |
| `n_envs` | 8 | Parallel envs (Colab SubprocVecEnv) |
| `total_timesteps` | 1,000,000 | Training budget |
| `policy` | MlpPolicy (256×256) | 2-layer MLP actor-critic |

---

### Training Setup (Google Colab)

```python
# Colab setup
!pip install mujoco gymnasium stable-baselines3

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_cb = CheckpointCallback(
    save_freq=100_000,
    save_path="/content/drive/MyDrive/leap_ckpts/",
    name_prefix="leap_ppo"
)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="/content/logs"
)

model.learn(total_timesteps=1_000_000, callback=checkpoint_cb)
model.save("leap_grasp_final")
```

> **Note:** Mount Google Drive before training to persist checkpoints across Colab session resets.

---

### Results

#### Learning Curve
![Learning Curve](assets/learning_curve.png)
> *[PLACEHOLDER — Replace with tensorboard export after training]*

#### Qualitative Demo
![Grasp Demo](assets/grasp_demo.gif)
> *[PLACEHOLDER — Replace with MuJoCo offscreen recording]*

#### Quantitative Metrics

| Metric | Value |
|---|---|
| Training steps | *[TBD]* |
| Mean episode reward (final 50k steps) | *[TBD]* |
| Grasp success rate (10 eval episodes) | *[TBD]* |
| Mean lift height achieved | *[TBD]* |
| Mean fingers in contact at lift | *[TBD]* |

---

### What Worked / What Didn't

> *[To be filled post-training — be honest here, this section is explicitly valued by the evaluators]*

**Worked:**
- *[e.g., approach reward converged quickly, contact reward effective for finger spreading]*

**Struggled:**
- *[e.g., lift reward sparse, needed curriculum: first stabilize approach, then enable lift]*

**Would do differently:**
- *[e.g., fingertip position in obs instead of palm centroid, asymmetric actor-critic]*

---

## Task B: ROS 2 Deployment Pipeline

### Node Architecture

```
┌─────────────────────┐        /hand/joint_commands        ┌──────────────────────┐
│    policy_node.py   │  ──────── Float64MultiArray ──────▶│  interface_node.py   │
│                     │                                     │                      │
│  Loads SB3 model    │                                     │  Converts to         │
│  Runs inference     │                                     │  sensor_msgs/        │
│  at 20Hz            │                                     │  JointState          │
│                     │                                     │  → /joint_states     │
└─────────────────────┘                                     └──────────┬───────────┘
                                                                       │
                                                            ┌──────────▼───────────┐
                                                            │  robot_state_        │
                                                            │  publisher           │
                                                            │  + RViz2             │
                                                            └──────────────────────┘
```

### Build & Launch

```bash
# On Ubuntu (ROS 2 Humble)
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

# Launch full pipeline
ros2 launch leap_hand_deploy leap_visualize.launch.py

# With trained policy
ros2 launch leap_hand_deploy leap_visualize.launch.py use_policy:=true model_path:=/path/to/leap_grasp_final.zip

# Fallback: sine wave trajectory (no policy required)
ros2 launch leap_hand_deploy leap_visualize.launch.py use_policy:=false
```

### Topics

| Topic | Type | Publisher | Subscriber |
|---|---|---|---|
| `/hand/joint_commands` | `std_msgs/Float64MultiArray` | policy_node | interface_node |
| `/joint_states` | `sensor_msgs/JointState` | interface_node | robot_state_publisher |
| `/tf` | `tf2_msgs/TFMessage` | robot_state_publisher | RViz2 |

### RViz2 Demo
![RViz2 Demo](assets/rviz_demo.gif)
> *[PLACEHOLDER — Replace with screen recording]*

---

## Environment Setup

### Local (Ubuntu 22.04 / ROS 2 Humble)

```bash
# Python dependencies
pip install mujoco gymnasium stable-baselines3 numpy matplotlib

# Clone menagerie for LEAP Hand model
git clone https://github.com/google-deepmind/mujoco_menagerie

# ROS 2 workspace
mkdir -p ros2_ws/src && cd ros2_ws
colcon build
source install/setup.bash
```

### Google Colab

Open `train/train_ppo.ipynb` in Colab. Set runtime to **GPU (T4)**. Mount Drive for checkpoint persistence.

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Dependencies

```
mujoco>=3.0.0
gymnasium>=0.29.0
stable-baselines3>=2.3.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.14.0
```

ROS 2: `Humble` | Ubuntu: `22.04`

---

## Submission Checklist

- [ ] `env/leap_grasp_env.py` — complete and tested
- [ ] `train/train_ppo.ipynb` — runnable on Colab
- [ ] Learning curve plot in `assets/`
- [ ] Grasp demo video/GIF in `assets/`
- [ ] ROS 2 package builds with `colcon build`
- [ ] RViz2 demo recording in `assets/`
- [ ] `report/report.pdf` — complete
- [ ] Google Form submitted

---

## Author

**Dhyan Dalwadi**
B.Tech ECE, IIIT Bhopal (2027)
Research focus: Perception-driven robotic manipulation, learning-based control
#   p o r t f o l i o _ w e b  
 