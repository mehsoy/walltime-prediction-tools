import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, RidgeCV
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

# Average CV score on the training set was:-467.1549141263793
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, min_samples_split=18)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.9500000000000001, tol=0.0001)),
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=14, p=1, weights="distance")),
    KNeighborsRegressor(n_neighbors=1, p=1, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
