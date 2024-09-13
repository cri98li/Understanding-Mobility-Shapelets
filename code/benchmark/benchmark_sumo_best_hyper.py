import os
import pickle
import random

import pandas as pd
from geoletrld.utils import Trajectories, y_from_df
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import readers as r
from tqdm.auto import tqdm

from benchmark_sumo import (build_model)
from run_model_from_transformations import evaluate_clf


def get_hyper_from_res(dataset) -> pd.DataFrame:
    df = pd.read_csv("results/best_hyper_cv.csv")

    return df[df.dataset_name == dataset]

def benchmark(df_base: pd.DataFrame, path_base="results/", dataset_name=""):
    df0 = df_base[["tid", "y", "time", "lat", "lon"]]

    random.seed(42)
    df = r.undersample(df0, 300)

    trajectories = Trajectories.from_DataFrame(df, latitude="lat", longitude="lon", time="time")
    y = y_from_df(df, tid_name="tid", y_name="y")


    X_train, X_test, y_train, y_test = train_test_split(list(trajectories), y, test_size=0.2, random_state=32,
                                                        stratify=y)

    X_train = Trajectories([(k, trajectories[k]) for i, k in enumerate(X_train)])
    X_test = Trajectories([(k, trajectories[k]) for i, k in enumerate(X_test)])

    hyper = get_hyper_from_res(dataset_name)

    for best_hyper in hyper.to_dict(orient="records"):
        tqdm.write(str(best_hyper))
        hyper_to_save = {
            "n_geo_factor": best_hyper["n_geo_factor"],
            "selection": best_hyper["selection"],
            "distance": best_hyper["distance"],
            "n_classes": best_hyper["n_classes"],
            "model": best_hyper["model"],
            "max_depth": best_hyper["max_depth"],
            "n_estimators": best_hyper["n_estimators"],
            "n_neighbors": best_hyper["n_neighbors"],
        }
        geo = build_model(None, best_hyper["n_geo_factor"],
                          best_hyper["selection"], best_hyper["distance"], best_hyper["n_classes"])

        if best_hyper["model"] == "DT":
            try:
                geo.model_to_fit = DecisionTreeClassifier(max_depth=int(best_hyper["max_depth"]) if not np.isnan(best_hyper["max_depth"]) else None, class_weight="balanced")
            except:
                geo.model_to_fit = DecisionTreeClassifier(max_depth=None, class_weight="balanced")
        elif best_hyper["model"] == "RF":
            geo.model_to_fit = RandomForestClassifier(n_estimators=int(best_hyper["n_estimators"]), class_weight="balanced", n_jobs=-1)
        elif best_hyper["model"] == "KNN":
            geo.model_to_fit = KNeighborsClassifier(n_neighbors=int(best_hyper["n_neighbors"]), n_jobs=-1)

        filename = f"{dataset_name}|{best_hyper['model']}"
        if os.path.exists(path_base + filename + ".pkl"):
            continue

        geo.fit(X_train, y_train)
        y_pred = geo.predict(X_test)
        scores_dict = evaluate_clf(y_test, y_pred, None)
        df = pd.DataFrame.from_dict([hyper_to_save | scores_dict])

        df.to_csv(path_base + filename + ".csv", index=False)
        pickle.dump(geo, open(path_base + filename + ".pkl", "wb"))


if __name__ == '__main__':
    readers = [
        r.sumo_cities,
        r.sumo_cities_grid_vs_borgo,
        r.sumo_cities_borgo_vs_mixed,
        r.sumo_cities_mixed_vs_grid,
        r.sumo_all_car_vs_bikes,
        r.sumo_cities_car_vs_bikes,
        r.sumo_grid_car_vs_bikes,
        r.sumo_borgo_car_vs_bikes,
        r.sumo_mixed_car_vs_bikes,
    ]

    progress_bar = tqdm(readers, position=0, leave=False)

    for reader in progress_bar:
        progress_bar.set_description(reader.__name__)
        df = reader()

        df_count = df.groupby("tid").size()
        df_count = df_count[(df_count > 60) & (df_count < 60 * 20)]

        df0 = df[df.tid.isin(df_count.keys())]

        benchmark(df0, path_base="results/final_models/", dataset_name=reader.__name__)
