import random

import pandas as pd

sumo_rename_map = {
    "vehicle_id": "tid",
    "class": "y"
}

def undersample(df, n_sample_per_class):
    classes = df.y.unique().tolist()
    selected_tid = []

    for classe in classes:
        selected_tid += random.sample(df[df.y == classe].tid.unique().tolist(), n_sample_per_class)

    return df[df.tid.isin(selected_tid)].sort_values(by=["tid", "time"])

def sumo_cities():
    df_grid = pd.read_csv("../generation/sumo_stuff/Grid-Empire-2.5km/trajectories.zip").rename(columns=sumo_rename_map)
    df_borgo = pd.read_csv("../generation/sumo_stuff/Borgo-Rome-2Km/trajectories.zip").rename(columns=sumo_rename_map)
    df_mixed = pd.read_csv("../generation/sumo_stuff/Mixed-Athens-2km/trajectories.zip").rename(columns=sumo_rename_map)

    df_grid.y = "grid"
    df_borgo.y = "borgo"
    df_mixed.y = "mixed"
    df_grid.tid = df_grid.tid.apply(lambda x: f"grid_{x}")
    df_borgo.tid = df_borgo.tid.apply(lambda x: f"borgo_{x}")
    df_mixed.tid = df_mixed.tid.apply(lambda x: f"mixed_{x}")

    return pd.concat([df_grid, df_borgo, df_mixed], ignore_index=True)

def sumo_cities_grid_vs_borgo():
    df_grid = pd.read_csv("../generation/sumo_stuff/Grid-Empire-2.5km/trajectories.zip").rename(columns=sumo_rename_map)
    df_borgo = pd.read_csv("../generation/sumo_stuff/Borgo-Rome-2Km/trajectories.zip").rename(columns=sumo_rename_map)

    df_grid.y = "grid"
    df_borgo.y = "borgo"
    df_grid.tid = df_grid.tid.apply(lambda x: f"grid_{x}")
    df_borgo.tid = df_borgo.tid.apply(lambda x: f"borgo_{x}")

    return pd.concat([df_grid, df_borgo], ignore_index=True)

def sumo_cities_borgo_vs_mixed():
    df_borgo = pd.read_csv("../generation/sumo_stuff/Borgo-Rome-2Km/trajectories.zip").rename(columns=sumo_rename_map)
    df_mixed = pd.read_csv("../generation/sumo_stuff/Mixed-Athens-2km/trajectories.zip").rename(columns=sumo_rename_map)

    df_borgo.y = "borgo"
    df_mixed.y = "mixed"
    df_borgo.tid = df_borgo.tid.apply(lambda x: f"borgo_{x}")
    df_mixed.tid = df_mixed.tid.apply(lambda x: f"mixed_{x}")

    return pd.concat([df_borgo, df_mixed], ignore_index=True)

def sumo_cities_mixed_vs_grid():
    df_grid = pd.read_csv("../generation/sumo_stuff/Grid-Empire-2.5km/trajectories.zip").rename(columns=sumo_rename_map)
    df_mixed = pd.read_csv("../generation/sumo_stuff/Mixed-Athens-2km/trajectories.zip").rename(columns=sumo_rename_map)

    df_grid.y = "grid"
    df_mixed.y = "mixed"
    df_grid.tid = df_grid.tid.apply(lambda x: f"grid_{x}")
    df_mixed.tid = df_mixed.tid.apply(lambda x: f"mixed_{x}")

    return pd.concat([df_grid, df_mixed], ignore_index=True)

def sumo_all_car_vs_bikes():
    df_grid = pd.read_csv("../generation/sumo_stuff/Grid-Empire-2.5km/trajectories.zip").rename(columns=sumo_rename_map)
    df_borgo = pd.read_csv("../generation/sumo_stuff/Borgo-Rome-2Km/trajectories.zip").rename(columns=sumo_rename_map)
    df_mixed = pd.read_csv("../generation/sumo_stuff/Mixed-Athens-2km/trajectories.zip").rename(columns=sumo_rename_map)

    df_grid.y = df_grid.y.apply(lambda x: f"grid_{x}")
    df_borgo.y = df_borgo.y.apply(lambda x: f"borgo_{x}")
    df_mixed.y = df_mixed.y.apply(lambda x: f"mixed_{x}")

    df_grid.tid = df_grid.tid.apply(lambda x: f"grid_{x}")
    df_borgo.tid = df_borgo.tid.apply(lambda x: f"borgo_{x}")
    df_mixed.tid = df_mixed.tid.apply(lambda x: f"mixed_{x}")

    return pd.concat([df_grid, df_borgo, df_mixed], ignore_index=True)

def sumo_cities_car_vs_bikes():
    df_grid = pd.read_csv("../generation/sumo_stuff/Grid-Empire-2.5km/trajectories.zip").rename(columns=sumo_rename_map)
    df_borgo = pd.read_csv("../generation/sumo_stuff/Borgo-Rome-2Km/trajectories.zip").rename(columns=sumo_rename_map)
    df_mixed = pd.read_csv("../generation/sumo_stuff/Mixed-Athens-2km/trajectories.zip").rename(columns=sumo_rename_map)

    df_grid.tid = df_grid.tid.apply(lambda x: f"grid_{x}")
    df_borgo.tid = df_borgo.tid.apply(lambda x: f"borgo_{x}")
    df_mixed.tid = df_mixed.tid.apply(lambda x: f"mixed_{x}")

    return pd.concat([df_grid, df_borgo, df_mixed], ignore_index=True)

def sumo_grid_car_vs_bikes():
    return pd.read_csv("../generation/sumo_stuff/Grid-Empire-2.5km/trajectories.zip").rename(columns=sumo_rename_map)

def sumo_borgo_car_vs_bikes():
    return pd.read_csv("../generation/sumo_stuff/Borgo-Rome-2Km/trajectories.zip").rename(columns=sumo_rename_map)

def sumo_mixed_car_vs_bikes():
    return pd.read_csv("../generation/sumo_stuff/Mixed-Athens-2km/trajectories.zip").rename(columns=sumo_rename_map)