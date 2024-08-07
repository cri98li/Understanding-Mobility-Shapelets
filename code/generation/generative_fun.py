from random import random

import numpy as np


def generate_trajectory_shape(n_steps, smoothness=11,
                              turning_angle=np.pi / 2,
                              starting_angle=np.pi / 2,
                              step_size=0.1,
                              noise_weight=.5):
    x, y = (random() - .5) * noise_weight, (random() - .5) * noise_weight
    trajectory = [(x, y)]

    smoothness_steps = 0

    for i in range(n_steps - 1):
        if i * 2 > n_steps - smoothness and smoothness_steps < smoothness:
            starting_angle -= turning_angle / smoothness
            smoothness_steps += 1

        x += step_size * np.cos(starting_angle)
        y += step_size * np.sin(starting_angle)
        trajectory.append((x, y))

    trajectory = np.array(trajectory)
    trajectory += (np.random.rand(*trajectory.shape) - .5) * noise_weight * step_size

    return trajectory
