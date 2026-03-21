import numpy as np
from stable_baselines3 import DQN
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


def evaluate_greedy(env, num_episodes=50):
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
            action = greedy_action(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        successes.append(info["success"])
        rewards.append(total_reward)
        steps_list.append(steps)
        distances.append(info["distance"])

    return {
        "success_rate": np.mean(successes) * 100,
        "avg_reward": np.mean(rewards),
        "avg_steps": np.mean(steps_list),
        "avg_distance": np.mean(distances),
    }


def evaluate_dqn(env, model, num_episodes=50):
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

    return {
        "success_rate": np.mean(successes) * 100,
        "avg_reward": np.mean(rewards),
        "avg_steps": np.mean(steps_list),
        "avg_distance": np.mean(distances),
    }


def print_results(name, results):
    print(f"\n===== {name} =====")
    print(f"Success Rate   : {results['success_rate']:.2f}%")
    print(f"Average Reward : {results['avg_reward']:.2f}")
    print(f"Average Steps  : {results['avg_steps']:.2f}")
    print(f"Average Dist   : {results['avg_distance']:.4f}")


if __name__ == "__main__":
    num_episodes = 50

    print("Evaluating Greedy Policy...")
    greedy_env = ArmEnv(render_mode=None)
    greedy_results = evaluate_greedy(greedy_env, num_episodes=num_episodes)
    greedy_env.close()

    print("Evaluating DQN Policy...")
    dqn_env = ArmEnv(render_mode=None)
    model = DQN.load("models/best_model/best_model")
    dqn_results = evaluate_dqn(dqn_env, model, num_episodes=num_episodes)
    dqn_env.close()

    print_results("Greedy Baseline", greedy_results)
    print_results("DQN Model", dqn_results)

    print("\n===== Comparison =====")
    print(f"Success Rate Gain   : {dqn_results['success_rate'] - greedy_results['success_rate']:.2f}%")
    print(f"Reward Gain         : {dqn_results['avg_reward'] - greedy_results['avg_reward']:.2f}")
    print(f"Step Reduction      : {greedy_results['avg_steps'] - dqn_results['avg_steps']:.2f}")
    print(f"Distance Reduction  : {greedy_results['avg_distance'] - dqn_results['avg_distance']:.4f}")