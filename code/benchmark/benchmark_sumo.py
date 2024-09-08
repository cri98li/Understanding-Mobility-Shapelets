import os.path
import pickle
import random

import numpy as np
import pandas as pd
from glob import glob

import psutil
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from geoletrld.model import Geolet
from geoletrld.selectors import RandomSelector, MutualInformationSelector, SelectorPipeline, ClusteringSelector, \
    GapSelector
from geoletrld.utils import Trajectories, y_from_df
from geoletrld.partitioners import NoPartitioner, GeohashPartitioner, FeaturePartitioner, SlidingWindowPartitioner
from geoletrld.distances import (EuclideanDistance, InterpolatedTimeDistance, LCSSTrajectoryDistance, FrechetDistance,
                                 CaGeoDistance, MatchComputeDistance)
from sklearn_extra.cluster import KMedoids
from tqdm.auto import tqdm

import readers as r
from readers import undersample

n_jobs = psutil.cpu_count(logical=True)

hyperparams = {
    "trj_class": [300],
    "n_geo_factor": [1, 2, 3, 5, 10, 20, 50, 100],
    "selection": [
        "random",
        "mi"
    ],
    "dist": [
        "ED_mean", "ED",
        #"ITD_mean", "ITD"
    ],
}

def build_model(trj_class, n_geo_factor, selection, dist, n_classes):
    k = n_geo_factor*n_classes

    if dist == "ED_mean":
        distance = EuclideanDistance(n_jobs=n_jobs, agg=np.mean, verbose=False)
    elif dist == "ITD_mean":
        distance = InterpolatedTimeDistance(n_jobs=n_jobs, agg=np.mean, verbose=False)
    elif dist == "ED":
        distance = EuclideanDistance(n_jobs=n_jobs, verbose=False)
    elif dist == "ITD":
        distance = InterpolatedTimeDistance(n_jobs=n_jobs, verbose=False)
    else:
        raise Exception("Unknown distance type")

    if selection == "random":
        selector = RandomSelector(k=k, verbose=False)
    elif selection == "mi":
        selector = SelectorPipeline(
            RandomSelector(k=k * 100, verbose=False),
            MutualInformationSelector(k=k, n_jobs=n_jobs, distance=distance, verbose=False),
        )
    else:
        raise Exception("Unknown selection type")

    return Geolet(
            partitioner=SlidingWindowPartitioner(window_size=30, verbose=False, overlap=5),
            selector=selector,
            distance=distance,
            model_to_fit=RandomForestClassifier(),
            verbose=False,
        )





def benchmark(df_base: pd.DataFrame, path_base="results/", dataset_name=""):
    n_splits = 10

    df0 = df_base[["tid", "y", "time", "lat", "lon"]]

    for trj_class in tqdm(hyperparams["trj_class"], position=1, leave=False, desc="trj_per_class"):
        df = undersample(df0, trj_class)

        trajectories = Trajectories.from_DataFrame(df, latitude="lat", longitude="lon", time="time")
        y = y_from_df(df, tid_name="tid", y_name="y")
        n_classes = len(np.unique(y))

        skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

        X_train, X_test, y_train, y_test = train_test_split(list(trajectories), y, test_size=0.2, random_state=32,
                                                            stratify=y)

        pickle.dump(X_train, open(path_base + f"Xtrain|{dataset_name}.pkl", "wb"))
        pickle.dump(X_test, open(path_base + f"Xtest|{dataset_name}.pkl", "wb"))
        pickle.dump(y_train, open(path_base + f"Ytrain|{dataset_name}.pkl", "wb"))
        pickle.dump(y_test, open(path_base + f"Ytest|{dataset_name}.pkl", "wb"))

        for i, (train_idx, validation_idx) in enumerate(tqdm(skf.split(X_train, y_train), position=2, leave=False, desc="cv_idx", total=n_splits)):
            if i != 0:
                continue

            X_train_cv = Trajectories([(k, trajectories[k]) for i, k in enumerate(X_train) if i in train_idx])
            X_validation = Trajectories([(k, trajectories[k]) for i, k in enumerate(X_train) if i in validation_idx])
            y_train_cv = y_train[train_idx]
            y_validation = y_train[validation_idx]

            for n_geo_factor in tqdm(hyperparams["n_geo_factor"], position=3, leave=False, desc="k_factor"):
                for selection in tqdm(hyperparams["selection"], position=4, leave=False, desc="Selection"):
                    for dist in tqdm(hyperparams["dist"], position=5, leave=False, desc="Distance"):
                        attr = [dataset_name, i, trj_class, n_geo_factor, n_geo_factor*n_classes, selection, dist,
                                n_classes]
                        filename = "|".join([str(x) for x in attr])
                        tqdm.write(filename)

                        if os.path.exists(path_base + f"yvalidationCV|{filename}.pkl"):
                            continue


                        classifier = build_model(trj_class, n_geo_factor, selection, dist, n_classes)
                        classifier.fit(X_train_cv, y_train_cv)
                        X_train_transf = classifier.transform(X_train_cv)
                        X_validation_transf = classifier.transform(X_validation)

                        pickle.dump(X_train_transf, open(path_base + f"XtrainCV|{filename}.pkl", "wb"))
                        pickle.dump(X_validation_transf, open(path_base + f"XvalidationCV|{filename}.pkl", "wb"))
                        pickle.dump(y_train_cv, open(path_base + f"ytrainCV|{filename}.pkl", "wb"))
                        pickle.dump(y_validation, open(path_base + f"yvalidationCV|{filename}.pkl", "wb"))


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

        benchmark(df0, path_base="results/transformations/", dataset_name=reader.__name__)