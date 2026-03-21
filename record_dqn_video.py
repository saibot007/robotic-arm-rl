import os
import imageio.v2 as imageio
from stable_baselines3 import DQN
from src.arm_env import ArmEnv

os.makedirs("videos", exist_ok=True)

env = ArmEnv(render_mode="rgb_array")
model = DQN.load("models/best_model/best_model")

frames = []

# try a seed that gives visible motion
obs, info = env.reset(seed=7)

# first frame
frame = env.render()
if frame is not None:
    frames.append(frame)

done = False
step_count = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    step_count += 1

env.close()

output_path = "videos/dqn_arm_manual.mp4"
imageio.mimsave(output_path, frames, fps=10)

print(f"Recorded steps: {step_count}")
print(f"Success: {info['success']}, final distance: {info['distance']:.4f}")
print(f"Saved video to: {output_path}")