import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-2880.827692982296
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=7, min_samples_leaf=2, min_samples_split=9)),
    SelectFwe(score_func=f_regression, alpha=0.021),
    MaxAbsScaler(),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=9, min_samples_leaf=19, min_samples_split=15)),
    SelectPercentile(score_func=f_regression, percentile=69),
    KNeighborsRegressor(n_neighbors=34, p=1, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
