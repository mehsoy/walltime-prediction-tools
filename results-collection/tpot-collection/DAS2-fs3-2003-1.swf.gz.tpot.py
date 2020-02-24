import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-310.1272159066585
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=6, min_samples_leaf=1, min_samples_split=12)),
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=19, p=2, weights="distance")),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=0.01, loss="ls", max_depth=5, max_features=0.05, min_samples_leaf=19, min_samples_split=15, n_estimators=100, subsample=0.6500000000000001)),
    KNeighborsRegressor(n_neighbors=1, p=2, weights="uniform")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
