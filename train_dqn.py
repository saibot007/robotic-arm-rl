import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from src.arm_env import ArmEnv

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

train_env = ArmEnv()
eval_env = ArmEnv()

train_env = gym.wrappers.RecordEpisodeStatistics(train_env)
eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best_model",
    log_path="./logs",
    eval_freq=1000,
    n_eval_episodes=20,
    deterministic=True,
    render=False,
)

model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=5e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.5,
    exploration_final_eps=0.05,
    verbose=1,
)

model.learn(total_timesteps=300000, callback=eval_callback)
model.save("models/dqn_robotic_arm_final")

train_env.close()
eval_env.close()

print("Training complete.")
print("Final model saved at: models/dqn_robotic_arm_final.zip")
print("Best model saved at: models/best_model/best_model.zip")