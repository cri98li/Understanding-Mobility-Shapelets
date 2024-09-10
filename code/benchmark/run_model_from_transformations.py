import pickle
from glob import glob
import os
import math
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import sklearn.metrics as skm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings

warnings.filterwarnings("ignore")

def evaluate_clf(y_test, y_pred, y_pred_proba):
    class_values = np.unique(y_test)
    binary = len(class_values) <= 2
    res = {
        'accuracy': skm.accuracy_score(y_test, y_pred),
        'balanced_accuracy': skm.balanced_accuracy_score(y_test, y_pred),
        'f1_score': skm.f1_score(y_test, y_pred, average='binary', pos_label=class_values[1])
        if binary else skm.f1_score(y_test, y_pred, average='micro'),
        'f1_micro': skm.f1_score(y_test, y_pred, average='micro'),
        'f1_macro': skm.f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': skm.f1_score(y_test, y_pred, average='weighted'),
        'precision_score': skm.precision_score(y_test, y_pred, average='binary', pos_label=class_values[1])
        if binary else skm.precision_score(y_test, y_pred, average='micro'),
        'precision_micro': skm.precision_score(y_test, y_pred, average='micro'),
        'precision_macro': skm.precision_score(y_test, y_pred, average='macro'),
        'precision_weighted': skm.precision_score(y_test, y_pred, average='weighted'),
        'recall_score': skm.recall_score(y_test, y_pred, average='binary', pos_label=class_values[1])
        if binary else skm.recall_score(y_test, y_pred, average='micro'),
        'recall_micro': skm.recall_score(y_test, y_pred, average='micro'),
        'recall_macro': skm.recall_score(y_test, y_pred, average='macro'),
        'recall_weighted': skm.recall_score(y_test, y_pred, average='weighted'),

    }

    if y_pred_proba is not None:
        res.update(
            {
                'roc_macro': skm.roc_auc_score(y_test, y_pred_proba[:, 1],
                                               average='macro') if binary else skm.roc_auc_score(
                    y_test, y_pred_proba, average='macro', multi_class='ovr'),
                'roc_micro': skm.roc_auc_score(y_test, y_pred_proba[:, 1],
                                               average='micro') if binary else skm.roc_auc_score(
                    y_test, y_pred_proba, average='micro', multi_class='ovr'),
                'roc_weighted': skm.roc_auc_score(y_test, y_pred_proba[:, 1], average='weighted')
                if binary else skm.roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr'),
                'average_precision_macro': skm.average_precision_score(y_test, y_pred_proba[:, 1], average='macro',
                                                                       pos_label=class_values[1])
                if binary else skm.average_precision_score(y_test, y_pred_proba, average='macro'),
                'average_precision_micro': skm.average_precision_score(y_test, y_pred_proba[:, 1], average='micro',
                                                                       pos_label=class_values[1])
                if binary else skm.average_precision_score(y_test, y_pred_proba, average='micro'),
                'average_precision_weighted': skm.average_precision_score(y_test, y_pred_proba[:, 1],
                                                                          average='weighted',
                                                                          pos_label=class_values[1])
                if binary else skm.average_precision_score(y_test, y_pred_proba, average='weighted'),
            }
        )

    return res

models = [
    ("DT-1", DecisionTreeClassifier, {'max_depth': 1, 'class_weight': 'balanced'}),
    ("DT-2", DecisionTreeClassifier, {'max_depth': 2, 'class_weight': 'balanced'}),
    ("DT-3", DecisionTreeClassifier, {'max_depth': 3, 'class_weight': 'balanced'}),
    ("DT-4", DecisionTreeClassifier, {'max_depth': 4, 'class_weight': 'balanced'}),
    ("DT-5", DecisionTreeClassifier, {'max_depth': 5, 'class_weight': 'balanced'}),
    ("DT-10", DecisionTreeClassifier, {'max_depth': 10, 'class_weight': 'balanced'}),
    ("DT-20", DecisionTreeClassifier, {'max_depth': 20, 'class_weight': 'balanced'}),
    ("DT-None", DecisionTreeClassifier, {'max_depth': None, 'class_weight': 'balanced'}),

    ("RF-100", RandomForestClassifier, {'n_estimators': 100, 'class_weight': 'balanced', 'n_jobs': 8}),
    ("RF-200", RandomForestClassifier, {'n_estimators': 200, 'class_weight': 'balanced', 'n_jobs': 8}),
    ("RF-500", RandomForestClassifier, {'n_estimators': 500, 'class_weight': 'balanced', 'n_jobs': 8}),
    ("RF-1000", RandomForestClassifier, {'n_estimators': 1000, 'class_weight': 'balanced', 'n_jobs': 8}),

    ("KNN-3", KNeighborsClassifier, {'n_neighbors': 3}),
    ("KNN-5", KNeighborsClassifier, {'n_neighbors': 5}),
    ("KNN-10", KNeighborsClassifier, {'n_neighbors': 10}),
    ("KNN-20", KNeighborsClassifier, {'n_neighbors': 10}),
]

if __name__ == "__main__":
    base_path = 'results/scores_cv/'

    dfs = []
    for X_train_path in tqdm(glob("results/transformations/XtrainCV*.pkl"), position=0, leave=False):
        X_validation_path = X_train_path.replace("XtrainCV", "XvalidationCV")
        y_train_path = X_train_path.replace("XtrainCV", "ytrainCV")
        y_validation_path = X_train_path.replace("XtrainCV", "yvalidationCV")

        X_train = pickle.load(open(X_train_path, "rb"))
        X_validation = pickle.load(open(X_validation_path, "rb"))
        y_train = pickle.load(open(y_train_path, "rb"))
        y_validation = pickle.load(open(y_validation_path, "rb"))

        attr = ["dataset_name", "cv_idx", "trj_class", "n_geo_factor", "n_geo_", "selection", "distance", "n_classes"]

        attr = {k:v for k,v in zip(attr, X_train_path.replace('.pkl', '').split("|")[1:])}

        for model_name, model_constructor, model_hyperparams in tqdm(models, position=1, leave=False):
            filename = model_name + '|' + '|'.join(X_train_path.split("|")[1:]).replace(".pkl", '.zip')
            if os.path.exists(base_path + filename):
                df = pd.read_csv(base_path + filename)
            else:

                clf = model_constructor(**model_hyperparams)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_validation)
                y_pred_proba = clf.predict_proba(X_validation)

                scores_dict = evaluate_clf(y_validation, y_pred, y_pred_proba)

                scores_dict.update(attr)
                scores_dict.update(model_hyperparams)
                scores_dict["model"] = model_name.split("-")[0]

                tqdm.write(f"{model_name}: {scores_dict['accuracy']*100:.2f}%")

                df = pd.DataFrame.from_dict([scores_dict])

                df.to_csv(base_path + filename, index=False)

            dfs.append(df)

    df = pd.concat(dfs)

    df.to_csv(base_path + "results.csv", index=False)