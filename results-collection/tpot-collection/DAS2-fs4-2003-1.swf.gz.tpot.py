import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import OneHotEncoder, StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-805.6529764814633
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=7, min_samples_leaf=4, min_samples_split=8)),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=9, min_samples_leaf=11, min_samples_split=19)),
    OneHotEncoder(minimum_fraction=0.2, sparse=False, threshold=10),
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.5, loss="square", n_estimators=100)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8, tol=0.001)),
    KNeighborsRegressor(n_neighbors=3, p=2, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
