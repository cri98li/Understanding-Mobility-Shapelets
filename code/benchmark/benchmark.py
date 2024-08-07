import pandas as pd
from glob import glob
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from geoletrld.model import Geolet
from geoletrld.selectors import RandomSelector, MutualInformationSelector, SelectorPipeline, ClusteringSelector, \
    GapSelector
from geoletrld.utils import Trajectories, y_from_df
from geoletrld.partitioners import NoPartitioner, GeohashPartitioner, FeaturePartitioner, SlidingWindowPartitioner
from geoletrld.distances import (EuclideanDistance, InterpolatedTimeDistance, LCSSTrajectoryDistance, FrechetDistance,
                                 CaGeoDistance, MatchComputeDistance)

def benchmark(df: pd.DataFrame):
    df = df[["tid", "y", "time", "lat", "lon"]]

    trajectories = Trajectories.from_DataFrame(df, latitude="lat", longitude="lon", time="time")
    y = y_from_df(df, tid_name="tid", y_name="y")

    X_train, X_test, y_train, y_test = train_test_split(list(trajectories), y, test_size=0.3, random_state=32,
                                                        stratify=y)

    X_train = Trajectories([(k, trajectories[k]) for k in X_train])
    X_test = Trajectories([(k, trajectories[k]) for k in X_test])

    classifier = Geolet(
        partitioner=NoPartitioner(),
        selector=SelectorPipeline(
            MutualInformationSelector(n_jobs=8, k=1, distance=InterpolatedTimeDistance(n_jobs=-1)),
            # GapSelector(k=5, n_jobs=10, distance=MatchComputeDistance(EuclideanDistance.best_fitting, LCSSTrajectoryDistance.best_fitting)),
            # ClusteringSelector(
            # KMeans(n_clusters=5), #è sbagliato?, ma funziona stranamente bene
            # KMedoids(n_clusters=30, metric='precomputed'),# n_jobs=9
            # AffinityPropagation(affinity="precomputed"), agg=lambda x: -np.sum(x),
            # OPTICS(metric="precomputed"),
            # SpectralClustering(affinity="precomputed"), agg=lambda x: -np.sum(x), #non gira
            # distance=LCSSTrajectoryDistance(n_jobs=10, verbose=True)
            # )
        ),
        distance=InterpolatedTimeDistance(n_jobs=8),
        model_to_fit=RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=32),
        # model_to_fit=KMeans(n_clusters=2)
    ).fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

def main():
    for filename in glob("../../data/*"):
        experiment_name = filename.split("/")[-1].split("\\")[-1].split(".")[0]
        df = pd.read_csv(filename)
        print(experiment_name)
        benchmark(df)
        print()

if __name__ == '__main__':
    main()