import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator, ZeroCount

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-3050.5593097031706
exported_pipeline = make_pipeline(
    ZeroCount(),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, min_samples_split=2)),
    RobustScaler(),
    GradientBoostingRegressor(alpha=0.75, learning_rate=0.1, loss="huber", max_depth=9, max_features=0.35000000000000003, min_samples_leaf=12, min_samples_split=7, n_estimators=100, subsample=0.8)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
