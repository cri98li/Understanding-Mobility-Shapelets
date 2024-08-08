import math
import pickle
import random

import numpy as np


def generate_trajectory_shape(n_steps, smoothness=11,
                              turning_angle=np.pi / 2,
                              starting_angle=np.pi / 2,
                              step_size=0.1,
                              noise_weight=.5):
    x, y = (random.random() - .5) * noise_weight, (random.random() - .5) * noise_weight
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


def yao_et_al_2017_sim_data(n_sample=10, speedPreSec=2, secPreCircle=20, a=100, b=10):
    ta = math.pi * 2
    noise = .3
    minLength = 70
    maxLength = 130
    minSample = 1
    maxSample = 2
    simData = []

    corrInCircleX = []
    corrInCircleY = []
    for i in range(secPreCircle):
        corrInCircleX.append(a * (math.sin(((i * 2 * math.pi) / secPreCircle) + 1.5 * math.pi) + 1))
        corrInCircleY.append(b * math.cos(((i * 2 * math.pi) / secPreCircle) + 0.5 * math.pi))

    # generate Straight
    for i in range(n_sample):
        seqTimeLength = random.randint(minLength, maxLength)
        sampleData = [[0, 0, 0]]
        j = 0
        previous = [0, 0, 0]
        while j < seqTimeLength:
            delta_t = random.randint(minSample, maxSample)
            x = previous[1] + random.gauss(delta_t * speedPreSec, noise)
            y = random.gauss(0, 1)
            j += delta_t
            sampleData.append([j, x, y])
            previous = [j, x, y]
        angle = random.random() * ta
        turnSample = []
        for point in sampleData:
            x = point[1]
            y = point[2]
            x1 = math.cos(angle) * x - math.sin(angle) * y
            y1 = math.cos(angle) * y + math.sin(angle) * x
            turnSample.append([point[0], x1, y1])
        simData.append(turnSample)

    # generate Circling
    for i in range(n_sample):
        seqTimeLength = random.randint(minLength, maxLength)
        sampleData = [[0, 0, 0]]
        j = 0
        while j < seqTimeLength:
            delta_t = random.randint(minSample, maxSample)
            x = random.gauss(corrInCircleX[j % secPreCircle], noise)
            y = random.gauss(corrInCircleY[j % secPreCircle], noise)
            j += delta_t
            sampleData.append([j, x, y])
        angle = random.random() * ta
        turnSample = []
        for point in sampleData:
            x = point[1]
            y = point[2]
            x1 = math.cos(angle) * x - math.sin(angle) * y
            y1 = math.cos(angle) * y + math.sin(angle) * x
            turnSample.append([point[0], x1, y1])
        simData.append(turnSample)

    # generate Bending
    for i in range(n_sample):
        seqTimeLength = random.randint(minLength, maxLength)
        sampleData = [[0, 0, 0]]
        j = 0
        previous = [0, 0, 0]
        while j < seqTimeLength:
            delta_t = random.randint(minSample, maxSample)
            x = previous[1] + random.gauss((delta_t * speedPreSec), noise)
            y = 500 * math.sin(j / (100 * math.pi))
            # random.gauss(0,50)
            j += delta_t
            sampleData.append([j, x, y])
            previous = [j, x, y]
        angle = random.random() * ta
        turnSample = []
        for point in sampleData:
            x = point[1]
            y = point[2]
            x1 = math.cos(angle) * x - math.sin(angle) * y
            y1 = math.cos(angle) * y + math.sin(angle) * x
            turnSample.append([point[0], x1, y1])
        simData.append(turnSample)

    for i in range(len(simData)):
        simData[i] = np.array(simData[i])

    for i in range(len(simData)):
        classe = "straight"
        if n_sample < i <= n_sample*2:
            classe = "circling"
        if i > n_sample*2:
            classe = "bending"
        arr = simData[i]
        arr = np.hstack([arr,
                         np.array(len(arr)*[classe]).reshape(-1, 1),
                         np.array(len(arr)*[i]).reshape(-1, 1)
                         ])
        simData[i] = arr

    sim_data_matrix = np.vstack(simData)

    return sim_data_matrix
