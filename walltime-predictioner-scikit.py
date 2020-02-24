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
# from helpers.tools import get_initial_df, get_all_workloads, fix_times, mean_absolute_percentage_error
# from helpers.tools import MASE, walltime_lambda_deviation, split_dates, get_tt_data, drop_large_workloads
import pandas as pd
import argparse
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt
import gc
import os,datetime, re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", default="chosen",
                        help="Filename to process in workloads/* dir: all / chosen / <filename>")
    parser.add_argument("--droplargeworkloads", action='store_true',
                        help="ignore large workloads")
    parser.add_argument("--droptrainsamplesbyseconds",
                        help="drop samples on training set, e.g., \
                             if runtime shorter than 60 (60 seconds)")
    parser.add_argument("--droptrainsamplesbyratio",
                        help="drop samples on training set, e.g., \
                             if runtime ratio smaller than 1 (1% of request walltime)")

    args = parser.parse_args()
    #opt = json.dumps(vars(args))
    return args

def get_automl_object(ml_algo):
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
        cls = MLPRegressor(verbose=True, max_iter=10)
        return cls
    if ml_algo == "SVR":
        cls = SVR()
        return cls

def drop_samples_by_ratio(X_train, y_train,ratio):
    ratios = (y_train / X_train['ReqWallTime']) * 100
    y_remove = (np.where(ratios < int(ratio)))[0].tolist()
    y_train = np.delete(y_train, y_remove)
    X_train = X_train.drop(X_train.index[y_remove])
    return X_train, y_train
def drop_samples_by_seconds(X_train, y_train,seconds):
    y_remove = (np.where(y_train < int(seconds)))[0].tolist()
    y_train = np.delete(y_train, y_remove)
    X_train = X_train.drop(X_train.index[y_remove])
    return X_train, y_train

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


    available_models = ["RandomForestRegressor",
                        "LinearRegression",
                        "SGDRegressor",
                        "MLPRegressor",
                        "DecisionTreeRegressor"
                        ]
    algo_header = {i: 1 for i in available_models}

    for workload in workloads:
        #read swf file into pandas dataframe
        print("Loading workload file:" + workload + "\n")
        df = get_initial_df(workload)
        #Drop fields cause these values are not know  for new sample
        df = df.drop('CPUTime',axis=1)
        df = df.drop('UsedMEM', axis=1)
        df = df.drop('Status', axis=1)
        #df = df.drop('JobId', axis=1)

        #Fix relative Suibmittime to absolute
        #and add Starttime of job (unixtimestamp)
        df = fix_times(workload,df)
        df = split_dates(df, "SubmitTime")
        df = split_dates(df, "StartTime")

        #Where are dropping original times
        #df contains now *_TS as numric timestamp fileds
        df = df.drop('StartTime', axis=1)
        df = df.drop('SubmitTime', axis=1)
        #change type of some columns / usefull for sklearn
        df['User'] = df['User'].astype('category')
        df['Group'] = df['Group'].astype('category')
        df['Class'] = df['Class'].astype('category')
        df['Partition'] = df['Partition'].astype('category')
        df['Exe'] = df['Exe'].astype('category')
        #dummy = df.to_numpy()


        y = df.pop('RunTime').values
        X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)
        #drop jobs smaller than
        #y_train = np.reshape(y_train, (-1, 1))

        if opt.droptrainsamplesbyseconds:
            seconds = opt.droptrainsamplesbyseconds
            X_train, y_train = drop_samples_by_seconds(X_train,y_train,seconds)
        elif opt.droptrainsamplesbyratio:
            ratio = opt.droptrainsamplesbyratio
            X_train, y_train = drop_samples_by_ratio(X_train, y_train, ratio)
            #y_remove = (np.where(y_train < int(opt.dropjobs)))[0].tolist()
            #y_train = np.delete(y_train, y_remove)
            #X_train = X_train.drop(X_train.index[y_remove])

        y_request = X_test['ReqWallTime'].values
        #y_request = np.reshape(y_request, (-1, 1))
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)



        df_cols = df.columns.tolist()
        # remove old dataframe and save memory
        del [[df]]
        gc.collect()

        #available_models = [ "RandomForestRegressor"]

        for algo in available_models:
            algodir = resultdir + "/" + algo
            figuredir = algodir + "/" + "figs"

            os.makedirs(algodir,exist_ok=True)
            os.makedirs(figuredir, exist_ok=True)

            #use_model = "MLPRegressor"
            model = get_automl_object(algo)
            model.fit(X_train, y_train)

            if algo in ['RandomForestRegressor' , 'DecisionTreeRegressor']:
                features = model.feature_importances_
                vip_list = list(zip(df_cols, model.feature_importances_))
                #print(model.sprint_statistics())
                #print(ml.show_models())
                #print(model.score(X_train, y_train))
                plt.bar(*zip(*vip_list))
                plt.xticks(rotation='vertical')
                plt.title(algo + " | " + workload)
                plt.tight_layout()
                figure = figuredir + "/" + "feature-importance-" + algo + "-" + workload + ".png"
                plt.savefig(figure)
                plt.close()
            y_predict = model.predict(X_test)
            train_score = float(model.score(X_train, y_train))
            test_score = float(model.score(X_test, y_test))
            mae = mean_absolute_error(y_test, y_predict)
            medae = median_absolute_error(y_test, y_predict)
            mase = MASE(y_train, y_test,y_predict)
            r2 = r2_score(y_test, y_predict)
            mape = mean_absolute_percentage_error(y_test, y_predict)
            wtld = walltime_lambda_deviation(y_test, y_predict, y_request)

            #Print the Metrics
            print("\nTrain score: " + str(train_score))
            print("Test score: " + str(test_score))
            print("MASE: " + str(mase))
            print("MAE: " + str(mae))
            print("MAPE: " + str(mape))
            print("lambda: " + str(wtld))
            print("MedAE: " + str(medae))

            resultfile = resultdir + "/" + "result-" + algo + ".txt"
            f = open(resultfile, "a")
            if algo_header[algo] == 1:
                f.write("#Workload;algorithm;train_score;test_score;mase;mae;medae;mape;lambda;r2")
                f.write("\n")
                algo_header[algo] = 0
            f.write(str(workload) + ";" + algo +
                    ";" + str(train_score) +
                    ":" + str(test_score) +
                    ":" + str(mase) +
                    ":" + str(mae) +
                    ":" + str(medae) +
                    ":" + str(mape) +
                    ":" + str(wtld) +
                    ":" + str(r2))
            f.write("\n")
            f.close
            #time.sleep(20)
            time.sleep(1)
    print("ende")


if __name__ == '__main__':
    main()
