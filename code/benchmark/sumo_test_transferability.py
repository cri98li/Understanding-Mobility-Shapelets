import pickle
import random

import pandas as pd
import numpy as np
import os
from glob import glob

from geoletrld.model import Geolet
from geoletrld.utils import Trajectories, y_from_df
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import readers as r
from run_model_from_transformations import evaluate_clf


def no_fit(geo, dest_dataset_X_test, dest_dataset_y_test):
    y_pred = geo.predict(dest_dataset_X_test)
    return evaluate_clf(y_pred, dest_dataset_y_test), geo

def refit_classifier_new_data(geo, dest_dataset_X_train, dest_dataset_y_train, dest_dataset_X_test, dest_dataset_y_test):
    geo.model_to_fit.fit(geo.transform(dest_dataset_X_train), dest_dataset_y_train)
    y_pred = geo.predict(dest_dataset_X_test)
    return evaluate_clf(y_pred, dest_dataset_y_test), geo

def refit_classifier_all_data(geo, source_dataset_X_train, source_dataset_y_train, source_dataset_X_test, source_dataset_y_test,
                              dest_dataset_X_train, dest_dataset_y_train, dest_dataset_X_test, dest_dataset_y_test):
    source_transformed = geo.transform(source_dataset_X_train)
    des_transformed = geo.transform(dest_dataset_X_train)

    X_train_all = np.append(source_transformed, des_transformed, axis=0)

    geo.model_to_fit.fit(X_train_all, np.append(source_dataset_y_train, dest_dataset_y_train), axis=0)
    y_pred_source = geo.predict(source_dataset_X_test)
    y_pred_dest = geo.predict(dest_dataset_X_test)
    y_pred_all = np.append(y_pred_source, y_pred_dest, axis=0)

    res_source = evaluate_clf(y_pred_source, source_dataset_y_test)
    res_dest = evaluate_clf(y_pred_dest, dest_dataset_y_test)
    res_all = evaluate_clf(y_pred_all, np.append(source_dataset_y_test, dest_dataset_y_test))


    return {f"source_{k}": v for k, v in res_source.items()} \
            | {f"dest_{k}": v for k, v in res_dest.items()} \
            | {f"all_{k}": v for k, v in res_all.items()}, geo

def preprocess(df0:pd.DataFrame):
    random.seed(42)
    df = r.undersample(df0, 300)

    trajectories = Trajectories.from_DataFrame(df, latitude="lat", longitude="lon", time="time")
    y = y_from_df(df, tid_name="tid", y_name="y")

    X_train, X_test, y_train, y_test = train_test_split(list(trajectories), y, test_size=0.2, random_state=32,
                                                        stratify=y)

    X_train = Trajectories([(k, trajectories[k]) for i, k in enumerate(X_train)])
    X_test = Trajectories([(k, trajectories[k]) for i, k in enumerate(X_test)])
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    datasets = {
        "sumo_all_car_vs_bikes": preprocess(r.sumo_all_car_vs_bikes()),
        "sumo_grid_car_vs_bikes": preprocess(r.sumo_grid_car_vs_bikes()),
        "sumo_borgo_car_vs_bikes": preprocess(r.sumo_borgo_car_vs_bikes()),
        "sumo_mixed_car_vs_bikes": preprocess(r.sumo_mixed_car_vs_bikes()),
    }

    for model_path in tqdm(glob('results/final_models/*.pkl'), position=0, leave=False):
        if origin_dataset_name not in datasets:
            continue

        geo = pickle.load(open(model_path, 'rb'))
        origin_dataset_name = model_path.split("/")[-1].split("|")[0]
        source_X_train, source_X_test, source_y_train, source_y_test = datasets[origin_dataset_name]
        for dataset_name, (dest_X_train, dest_X_test, dest_y_train, dest_y_test) \
                in tqdm(datasets.items(), position=1, leave=False):
            model_name = model_path.split("/")[-1].replace(".pkl", "")

            if dataset_name == origin_dataset_name:
                continue

            if not os.path.exists(f"results/transfered_models/{model_name}|{dataset_name}_no.pkl"):
                res, new_geo = no_fit(geo, dest_X_test, dest_y_test)
                pd.DataFrame.from_dict([res]).to_csv(f"results/transfered_models/{model_name}|{dataset_name}_no.pkl", index=False)
                pickle.dump(new_geo, open(f"results/transfered_models/{model_name}|{dataset_name}_no.pkl", "wb"))

            if not os.path.exists(f"results/transfered_models/{model_name}|{dataset_name}_partial.pkl"):
                res, new_geo = refit_classifier_new_data(geo, dest_X_train, dest_y_train, dest_X_test, dest_y_test)
                pd.DataFrame.from_dict([res]).to_csv(f"results/transfered_models/{model_name}|{dataset_name}_partial.pkl", index=False)
                pickle.dump(new_geo, open(f"results/transfered_models/{model_name}|{dataset_name}_partial.pkl", "wb"))

            if not os.path.exists(f"results/transfered_models/{model_name}|{dataset_name}_all.pkl"):
                res, new_geo = refit_classifier_new_data(geo, source_X_train, source_y_train, source_X_test, source_y_test,
                                                         dest_X_train, dest_y_train, dest_X_test, dest_y_test)
                pd.DataFrame.from_dict([res]).to_csv(f"results/transfered_models/{model_name}|{dataset_name}_all.pkl", index=False)
                pickle.dump(new_geo, open(f"results/transfered_models/{model_name}|{dataset_name}_all.pkl", "wb"))

