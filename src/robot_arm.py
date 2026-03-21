import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

L1 = 1
L2 = 1

theta1 = 0.5
theta2 = 0.5

target_x = 1.5
target_y = 1.5

plt.ion()
fig, ax = plt.subplots()

for step in range(100):

    # forward kinematics
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)

    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)

    # distance from target
    error_x = target_x - x2
    error_y = target_y - y2
    distance = np.sqrt(error_x**2 + error_y**2)

    print(f"Step {step}: End Effector=({x2:.2f}, {y2:.2f}), Distance={distance:.2f}")

    # very simple manual control rule
    if error_x > 0:
        theta1 += 0.02
    else:
        theta1 -= 0.02

    if error_y > 0:
        theta2 += 0.02
    else:
        theta2 -= 0.02

    # plotting
    ax.clear()
    ax.plot([0, x1], [0, y1], marker='o')
    ax.plot([x1, x2], [y1, y2], marker='o')
    ax.scatter(target_x, target_y, color='red', s=80)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title("Robotic Arm Moving Toward Target")
    ax.grid()

    plt.pause(0.05)

    if distance < 0.1:
        print("Target reached!")
        break

plt.ioff()
plt.show()