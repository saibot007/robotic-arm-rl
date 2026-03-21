import numpy as np
from stable_baselines3 import DQN
from src.arm_env import ArmEnv

env = ArmEnv(render_mode=None)
model = DQN.load("models/best_model/best_model")

num_episodes = 50
successes = []
rewards = []
steps_list = []
distances = []

for ep in range(num_episodes):
    obs, info = env.reset(seed=ep)
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    successes.append(info["success"])
    rewards.append(total_reward)
    steps_list.append(steps)
    distances.append(info["distance"])

    print(
        f"Episode {ep+1}/{num_episodes} | "
        f"Reward: {total_reward:.2f} | "
        f"Steps: {steps} | "
        f"Success: {info['success']} | "
        f"Distance: {info['distance']:.4f}"
    )

print("\n===== Best Model Evaluation =====")
print(f"Success Rate   : {np.mean(successes) * 100:.2f}%")
print(f"Average Reward : {np.mean(rewards):.2f}")
print(f"Average Steps  : {np.mean(steps_list):.2f}")
print(f"Average Dist   : {np.mean(distances):.4f}")

env.close()