#!/usr/bin/env python3
# Copyright 2019 Mehmet Soysal(mehmet.soysal@kit.edu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from helpers.tools import *
#from helpers.tools import MASE, walltime_lambda_deviation, split_dates, get_tt_data, drop_large_workloads
import pandas as pd
import argparse
import time
from sklearn.model_selection import train_test_split
from autosklearn import regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
import sklearn
from tpot import TPOTRegressor
import autosklearn.metrics
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn import tree
import matplotlib.pyplot as plt
import gc, os, datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", default="chosen",
                        help="Filename to process in workloads/* dir: all / chosen / <filename>")
    parser.add_argument("--droplargeworkloads", action='store_true',
                        help="ignore large workloads")

    args = parser.parse_args()
    #opt = json.dumps(vars(args))
    return args

def get_automl_object(ml_algo):
    """
    returns a ml model which can be trained
    :param ml_algo: string which model schoulkd be returned
    :return:
    """
    cls = ""
    if ml_algo == "SGDRegressor":
        cls = sklearn.linear_model.SGDRegressor()
        return cls
    if ml_algo == "PassiveAggressiveRegressor":
        cls = sklearn.linear_model.PassiveAggressiveRegressor()
        return cls
    if ml_algo == "KNeighborsRegressor":
        cls = KNeighborsRegressor()
        return cls
    if ml_algo == "LinearRegression":
        cls = sklearn.linear_model.LinearRegression()
        return cls
    if ml_algo == "RandomForestRegressor":
        cls = RandomForestRegressor()
        return cls
    if ml_algo == "DecisionTreeRegressor":
        cls = DecisionTreeRegressor()
        return cls
    if ml_algo == "ExtraTreeRegressor":
        cls = ExtraTreeRegressor()
        return cls
    if ml_algo == "MLPRegressor":
        cls = MLPRegressor(verbose=True)
        return cls
    if ml_algo == "SVR":
        cls = SVR()
        return cls
    if ml_algo == "tpot":
        cls = TPOTRegressor(generations = 1, population_size=1, verbosity=2, scoring="neg_mean_absolute_error",n_jobs=4)
        return cls
    if ml_algo == "sklearn":
        runtime_seconds = 60
        per_task = 60
        mem_limit = 8000
        #seconmds
        #cls = regression.AutoSklearnRegressor(include_estimators=["random_forest" , "decision_tree" , "extra_trees"  ], time_left_for_this_task=60)
        cls = regression.AutoSklearnRegressor(#time_left_for_this_task=runtime_seconds,
                                              #per_run_time_limit = per_task,
                                              n_jobs=4,
                                              ensemble_nbest=5,
                                              ml_memory_limit=mem_limit
                                              #include_estimators = ["sgd"]
                                              )
                                             #   include_estimators = ["decision_tree"]
                                             #   )

        return cls



def main():
    opt = parse_args()
    droplargeworkloads = 1
    workloads = []
    if opt.filename:
        if opt.filename == "all":
            workloads = get_all_workloads()
        elif opt.filename == "chosen":
            workloads = ["fh1_workload.swf.gz",
                         "CTC-SP2-1996-3.1-cln.swf.gz",
                         "KTH-SP2-1996-2.1-cln.swf.gz",
                         "SDSC-Par-1996-3.1-cln.swf.gz",
                         "KIT-FH2-2016-1.swf.gz"]
        else:
            workloads = [opt.filename]
    if opt.droplargeworkloads:
        workloads = drop_large_workloads(workloads)
    workloads.sort()

    #create result dir
    resultdir = os.path.join("results/", os.uname()[1], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(resultdir, exist_ok=True)


    available_models = ["tpot"]
    algo_header = {i: 1 for i in available_models}

    for workload in workloads:
        #read swf file into pandas dataframe
        print(workload)
        df = get_initial_df(workload)
        df = fix_times(workload, df)
        df = split_dates(df, "SubmitTime")
        df = split_dates(df, "StartTime")

        #Where are dropping original times
        #df contains now *_TS as numric timestamp fileds
        df = df.drop('StartTime', axis=1)
        df = df.drop('SubmitTime', axis=1)
        X_train, X_test, y_train, y_test = get_tt_data(df)
        y_request = X_test['ReqWallTime'].values


        #Get dtype of column (integer and categorical)
        dtypes = []
        #print(df.dtypes)
        dummy = df.dtypes.tolist()
        df_cols = df.columns.tolist()
        for col in df_cols:
            if df[col].dtype.kind in 'bifc':
                dtypes.append("Numerical")
            else:
                dtypes.append("Categorical")

        #remove old dataframe and save memory
        del [[df]]
        gc.collect()


        for algo in available_models:
            algodir = resultdir + "/" + algo
            figuredir = algodir + "/" + "figs"

            os.makedirs(algodir,exist_ok=True)
            os.makedirs(figuredir, exist_ok=True)

            #get model to train and fit
            model = get_automl_object(algo)
            if model == "sklearn":
                #, metric=autosklearn.metrics.r2
                model.fit(X_train, y_train, feat_type=dtypes, metric=autosklearn.metrics.mean_absolute_error)
                print(model.show_models())
                print(model.sprint_statistics())
            elif model == "tpot":
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            #prediction and metrics
            y_predict = model.predict(X_test)
            train_score = float(model.score(X_train, y_train))
            test_score = float(model.score(X_test, y_test))
            mae = mean_absolute_error(y_test, y_predict)
            medae = median_absolute_error(y_test, y_predict)
            mase = MASE(y_train, y_test,y_predict)
            r2 = r2_score(y_test, y_predict)
            mape = mean_absolute_percentage_error(y_test, y_predict)
            wtld = walltime_lambda_deviation(y_test, y_predict, y_request)

            #Print to stdout
            print("\nTrain score: " + str(train_score))
            print("Test score: " + str(test_score))
            print("MASE: " + str(mase))
            print("MAE: " + str(mae))
            print("MAPE: " + str(mape))
            print("lambda: " + str(wtld))
            print("MedAE: " + str(medae))


            #write results to file output
            result_file = '{}/result-{}.txt'.format(resultdir, algo)
            with open(result_file, 'a') as output_handle:
                #first write a headline
                if algo_header[algo] == 1:
                    output_handle.write("#Workload;algorithm;train_score;test_score;mase;mae;medae;mape;lambda;r2")
                    output_handle.write("\n")
                    algo_header[algo] = 0
                output_handle.write('{};{};{};{};{};{};{};{};{};{}\n'.format(
                    workload, algo, train_score, test_score, mase, mae, medae, mape, wtld, r2
                ))


            #IF using tpot save trained model
            if algo == "tpot":
                model.export(resultdir + "/" + workload + ".tpot.py")
            #calm down
            time.sleep(1)
    print("Everything done - HAVE A NICE DAY")


if __name__ == '__main__':
    main()
