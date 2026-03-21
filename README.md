# 2D Robotic Arm with Reinforcement Learning

A custom reinforcement learning project where a 2-link planar robotic arm learns to reach randomly generated target points in 2D space.

## Project Overview

This project implements a 2D robotic arm simulation using:

- Python
- NumPy
- Matplotlib
- Gymnasium
- Stable-Baselines3

The arm is modeled as a 2-link manipulator with discrete joint actions. A custom Gymnasium environment was built from scratch, and a DQN agent was trained to control the arm to reach target positions.

## What the Arm Does

In each episode:

- the arm starts from an initial configuration
- a random reachable target is generated
- the agent selects joint actions
- the arm moves until it reaches the target or the step limit is reached

The goal is to learn an efficient target-reaching policy.

## Action Space

The agent uses 4 discrete actions:

- `0` → increase joint 1 angle
- `1` → decrease joint 1 angle
- `2` → increase joint 2 angle
- `3` → decrease joint 2 angle

## Observation Space

The state includes:

- joint angles
- target position
- end-effector position
- relative offset between target and end-effector

## Implemented Components

- Forward kinematics for a 2-link robotic arm
- Matplotlib-based 2D visualization
- Custom Gymnasium environment
- Reward shaping for target reaching
- Greedy baseline controller
- DQN training with Stable-Baselines3
- Multi-episode evaluation
- Greedy vs DQN comparison
- Video recording of trained rollouts

## Results

Final comparison over 50 evaluation episodes:

### Greedy Baseline
- Success Rate: **72.00%**
- Average Reward: **46.06**
- Average Steps: **109.60**
- Average Distance: **0.1606**

### DQN Model
- Success Rate: **76.00%**
- Average Reward: **49.64**
- Average Steps: **83.52**
- Average Distance: **0.1141**

## Key Outcome

The trained DQN agent outperformed the greedy baseline in:

- success rate
- efficiency
- final target accuracy

## Project Structure

```bash
robotic_arm_rl/
├── src/
│   └── arm_env.py
├── models/
├── logs/
├── videos/
├── check_env.py
├── main.py
├── train_dqn.py
├── evaluate_dqn.py
├── compare_agents.py
├── record_dqn_video.py
├── requirements.txt
└── README.md
```

 ## Important Note

This project is a *robotic arm simulation and reinforcement learning environment*, not a full digital twin.

A true digital twin would require a real physical robotic system, sensor/state synchronization, and a live link between the physical and virtual system. This project is currently a simulation-based control and learning setup, which can later be extended toward digital twin style workflows.