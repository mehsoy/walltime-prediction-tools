import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-3474.190419690249
exported_pipeline = make_pipeline(
    StandardScaler(),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=8, min_samples_leaf=4, min_samples_split=12)),
    Nystroem(gamma=0.55, kernel="poly", n_components=7),
    KNeighborsRegressor(n_neighbors=43, p=2, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
