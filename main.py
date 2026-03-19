import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

num_episodes = 5
all_scores = []

for episode in range(num_episodes):
    obs, info = env.reset()
    steps_survived = 0

    while True:
        pole_angle = obs[2]

        # simple rule-based logic
        if pole_angle > 0:
            action = 1
        else:
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        steps_survived += 1

        if terminated or truncated:
            print(f"Episode {episode + 1}: {steps_survived} steps")
            all_scores.append(steps_survived)
            break

average_score = sum(all_scores) / len(all_scores)
print("\nAll scores:", all_scores)
print("Average steps survived:", average_score)

env.close()
with open("results.txt", "w") as f:
    f.write(f"Scores: {all_scores}\n")
    f.write(f"Average: {average_score}\n")