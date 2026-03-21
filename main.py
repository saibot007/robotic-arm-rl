import numpy as np
from src.arm_env import ArmEnv


def greedy_action(env, obs):
    best_action = None
    best_score = -float("inf")

    theta1, theta2 = env.theta1, env.theta2

    for action in range(env.action_space.n):
        t1, t2 = theta1, theta2

        if action == 0:
            t1 += env.step_size
        elif action == 1:
            t1 -= env.step_size
        elif action == 2:
            t2 += env.step_size
        elif action == 3:
            t2 -= env.step_size

        ee = env.forward_kinematics(t1, t2)
        distance = np.linalg.norm(env.target - ee)

        score = -distance
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def evaluate(env, num_episodes=50, render=False):
    rewards = []
    steps_list = []
    successes = []
    distances = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = greedy_action(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            if render:
                env.render()

        rewards.append(total_reward)
        steps_list.append(steps)
        successes.append(info["success"])
        distances.append(info["distance"])

        print(
            f"Episode {ep+1}/{num_episodes} | "
            f"Reward: {total_reward:.2f} | "
            f"Steps: {steps} | "
            f"Success: {info['success']} | "
            f"Distance: {info['distance']:.4f}"
        )

    print("\n===== Summary =====")
    print(f"Success rate : {100 * np.mean(successes):.2f}%")
    print(f"Avg reward   : {np.mean(rewards):.2f}")
    print(f"Avg steps    : {np.mean(steps_list):.2f}")
    print(f"Avg dist     : {np.mean(distances):.4f}")


if __name__ == "__main__":
    env = ArmEnv(render_mode=None)
    evaluate(env, num_episodes=50, render=False)
    env.close()