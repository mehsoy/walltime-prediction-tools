import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-3670.8320170190973
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=65),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=10, min_samples_leaf=3, min_samples_split=14)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.05, min_samples_leaf=7, min_samples_split=2, n_estimators=100)),
    DecisionTreeRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=9)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
