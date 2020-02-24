import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-542.0157820577372
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=DecisionTreeRegressor(max_depth=6, min_samples_leaf=11, min_samples_split=15)),
        StackingEstimator(estimator=make_pipeline(
            make_union(
                FunctionTransformer(copy),
                make_pipeline(
                    FeatureAgglomeration(affinity="l2", linkage="complete"),
                    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=34, p=2, weights="distance")),
                    StackingEstimator(estimator=LinearSVR(C=0.001, dual=True, epsilon=0.0001, loss="epsilon_insensitive", tol=1e-05)),
                    StandardScaler()
                )
            ),
            StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.75, tol=0.1)),
            DecisionTreeRegressor(max_depth=3, min_samples_leaf=4, min_samples_split=18)
        ))
    ),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.7000000000000001, tol=0.1)),
    KNeighborsRegressor(n_neighbors=11, p=2, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
