import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import OneHotEncoder, StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-5557.24844865042
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        make_pipeline(
            StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.5, loss="ls", max_depth=1, max_features=0.45, min_samples_leaf=17, min_samples_split=12, n_estimators=100, subsample=0.9000000000000001)),
            StackingEstimator(estimator=LassoLarsCV(normalize=True)),
            StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=2, p=1, weights="distance")),
            OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
            OneHotEncoder(minimum_fraction=0.05, sparse=False, threshold=10)
        )
    ),
    KNeighborsRegressor(n_neighbors=18, p=1, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
