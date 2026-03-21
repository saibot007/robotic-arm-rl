import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class ArmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode=None):
        super().__init__()

        self.link1 = 1.0
        self.link2 = 1.0

        self.step_size = 0.05
        self.max_steps = 150
        self.success_threshold = 0.08

        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)

        low = np.array(
            [-np.pi, -np.pi, -2.0, -2.0, -2.0, -2.0, -4.0, -4.0],
            dtype=np.float32
        )
        high = np.array(
            [np.pi, np.pi, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.fig = None
        self.ax = None

        self.theta1 = 0.0
        self.theta2 = 0.0
        self.steps = 0
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self._previous_distance = None

    def forward_kinematics(self, theta1, theta2):
        x1 = self.link1 * np.cos(theta1)
        y1 = self.link1 * np.sin(theta1)

        x2 = x1 + self.link2 * np.cos(theta1 + theta2)
        y2 = y1 + self.link2 * np.sin(theta1 + theta2)

        return np.array([x2, y2], dtype=np.float32)

    def _get_obs(self):
        ee = self.forward_kinematics(self.theta1, self.theta2)
        dx = self.target[0] - ee[0]
        dy = self.target[1] - ee[1]

        return np.array([
            self.theta1,
            self.theta2,
            self.target[0],
            self.target[1],
            ee[0],
            ee[1],
            dx,
            dy
        ], dtype=np.float32)

    def _get_info(self):
        ee = self.forward_kinematics(self.theta1, self.theta2)
        distance = np.linalg.norm(self.target - ee)
        success = bool(distance < self.success_threshold)

        return {
            "distance": float(distance),
            "success": success,
            "is_success": success
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.theta1 = 0.0
        self.theta2 = 0.0
        self.steps = 0

        angle = self.np_random.uniform(-np.pi, np.pi)
        radius = self.np_random.uniform(0.2, self.link1 + self.link2 - 0.1)

        self.target = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle)
        ], dtype=np.float32)

        obs = self._get_obs()
        info = self._get_info()
        self._previous_distance = info["distance"]

        return obs, info

    def step(self, action):
        if action == 0:
            self.theta1 += self.step_size
        elif action == 1:
            self.theta1 -= self.step_size
        elif action == 2:
            self.theta2 += self.step_size
        elif action == 3:
            self.theta2 -= self.step_size

        self.theta1 = ((self.theta1 + np.pi) % (2 * np.pi)) - np.pi
        self.theta2 = ((self.theta2 + np.pi) % (2 * np.pi)) - np.pi

        self.steps += 1

        obs = self._get_obs()
        info = self._get_info()
        curr_distance = info["distance"]

        terminated = info["success"]
        truncated = self.steps >= self.max_steps

        reward = 0.0
        reward += (self._previous_distance - curr_distance) * 12.0
        reward -= 0.01
        reward -= 0.05 * curr_distance

        if terminated:
            reward += 40.0

        self._previous_distance = curr_distance

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()

        x1 = self.link1 * np.cos(self.theta1)
        y1 = self.link1 * np.sin(self.theta1)

        x2 = x1 + self.link2 * np.cos(self.theta1 + self.theta2)
        y2 = y1 + self.link2 * np.sin(self.theta1 + self.theta2)

        self.ax.plot([0, x1, x2], [0, y1, y2], marker="o", linewidth=3)
        self.ax.scatter(self.target[0], self.target[1], marker="x", s=150)

        self.ax.set_xlim(-2.2, 2.2)
        self.ax.set_ylim(-2.2, 2.2)
        self.ax.set_aspect("equal")
        self.ax.set_title(f"2D Robotic Arm | Step {self.steps}")

        self.fig.canvas.draw()

        if self.render_mode == "human":
            plt.pause(0.05)
            return None

        if self.render_mode == "rgb_array":
            width, height = self.fig.canvas.get_width_height()
            image = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape((height, width, 4))
            return image[:, :, :3].copy()


        return None

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None