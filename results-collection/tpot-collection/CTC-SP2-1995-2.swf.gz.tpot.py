import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-6272.049750287444
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=8, min_samples_split=10, n_estimators=100)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.45, tol=0.0001)),
    GradientBoostingRegressor(alpha=0.75, learning_rate=0.5, loss="lad", max_depth=5, max_features=0.25, min_samples_leaf=4, min_samples_split=18, n_estimators=100, subsample=0.7500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
