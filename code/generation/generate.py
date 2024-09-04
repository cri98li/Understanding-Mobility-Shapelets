import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generative_fun import generate_trajectory_shape, yao_et_al_2017_sim_data


def prepare_data_shape(n_trj_per_shape, shapes: list = None):
    random.seed(42)
    if shapes is None:
        shapes = [
            np.pi / 2,  # turn right
            0
        ]

    trajectories = []
    for shape_idx, shape in enumerate(shapes):
        for i in range(n_trj_per_shape):
            trj = np.hstack([np.ones((100, 1)) * shape_idx,
                             generate_trajectory_shape(n_steps=100,
                                                       smoothness=20,
                                                       turning_angle=shape,
                                                       step_size=0.1,
                                                       noise_weight=.2)])
            trajectories.append(trj)

    for i in range(len(trajectories)):
        trajectories[i] = np.hstack(
            [trajectories[i], np.arange(0, 100, dtype=int).reshape(100, 1), np.ones((100, 1)) * i])

    df = pd.DataFrame(np.vstack(trajectories), columns=["y", "lat", "lon", "time", "tid"])

    y_rename = [
        "right",
        "straight"
    ]
    df.y = df.y.astype(int).apply(lambda x: y_rename[x])

    return df.astype({
        "time": np.int_,
        "tid": np.int_,
    })


def prepare_data_time(n_trj_per_shape, shapes: list = None):
    random.seed(42)
    if shapes is None:
        shapes = [
            np.pi / 2,  # turn right
            np.pi / 2
        ]

    trajectories = []
    for shape_idx, shape in enumerate(shapes):
        for i in range(n_trj_per_shape):
            trj = np.hstack([np.ones((100, 1)) * shape_idx,
                             generate_trajectory_shape(n_steps=100,
                                                       smoothness=20,
                                                       turning_angle=shape,
                                                       step_size=0.1,
                                                       noise_weight=.2)])
            trajectories.append(trj)

    for i in range(len(trajectories)):
        trajectories[i] = np.hstack(
            [trajectories[i],
             np.arange(0, 100, dtype=int).reshape(100, 1),
             np.ones((100, 1)) * i])

        trajectories[i][:, -2] *= 1 if i < len(trajectories) / 2 else 5

    df = pd.DataFrame(np.vstack(trajectories), columns=["y", "time", "lat", "lon", "tid"])

    y_rename = [
        "fast",
        "slow"
    ]
    df.y = df.y.astype(int).apply(lambda x: y_rename[x])

    return df.astype({
        "time": np.int_,
        "tid": np.int_,
    })


def prepare_data_shape_time(n_trj_per_shape, shapes: list = None):
    random.seed(42)
    if shapes is None:
        shapes = [
            np.pi / 2,  # turn right
            0,
            np.pi / 2,
            0
        ]

    trajectories = []
    for shape_idx, shape in enumerate(shapes):
        for i in range(n_trj_per_shape):
            trj = np.hstack([np.ones((100, 1)) * shape_idx,
                             generate_trajectory_shape(n_steps=100,
                                                       smoothness=20,
                                                       turning_angle=shape,
                                                       step_size=0.1,
                                                       noise_weight=.2)])
            trajectories.append(trj)

    for i in range(len(trajectories)):
        trajectories[i] = np.hstack(
            [trajectories[i],
             np.arange(0, 100, dtype=int).reshape(100, 1),
             np.ones((100, 1)) * i])

        trajectories[i][:, -2] *= 1 if i < len(trajectories) / 2 else 5

    df = pd.DataFrame(np.vstack(trajectories), columns=["y", "lat", "lon", "time", "tid"])

    y_rename = [
        "fast_right",
        "fast_straight",
        "slow_right",
        "slow_straight"
    ]
    df.y = df.y.astype(int).apply(lambda x: y_rename[x])

    return df.astype({
        "time": np.int_,
        "tid": np.int_,
    })


def prepare_data_yao_et_al_2017_sim_data(n_sample=100):
    arr = yao_et_al_2017_sim_data(n_sample=n_sample)

    df = pd.DataFrame(arr, columns=["time", "lat", "lon", "y", "tid"]).infer_objects()
    df.time = df.time.apply(lambda x: round(float(x)))

    return df.astype({
        "lat": np.float_,
        "lon": np.float_,
        "time": np.int_,
        "tid": np.int_,
    })


def plot_data(df, trj_for_class=2):
    for y in df.y.unique():
        for tid in df[df.y == y].tid.unique()[:trj_for_class]:
            plt.plot(df[df.tid == tid].lat.tolist(), df[df.tid == tid].lon.tolist(), label=f"{tid}_{y}")

    plt.legend()
    plt.show()


def main():
    df = prepare_data_shape(100)
    plot_data(df)
    df.to_csv("../../data/simple_shape.zip", index=False)

    df = prepare_data_time(100)
    plot_data(df)
    df.to_csv("../../data/simple_time.zip", index=False)

    df = prepare_data_shape_time(50)
    plot_data(df)
    df.to_csv("../../data/simple_shape_time.zip", index=False)

    df = prepare_data_yao_et_al_2017_sim_data()
    plot_data(df)
    df.to_csv("../../data/simple_yao.zip", index=False)


if __name__ == '__main__':
    main()
