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
import gc, os, datetime
from progressbar import ProgressBar
import argparse
import time
import pandas as pd
import numpy as np

from tpot import TPOTRegressor
from autosklearn import regression
import autosklearn.metrics

import sklearn
from sklearn import tree, ensemble,neural_network, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_absolute_error
from sklearn.externals.six import StringIO




import pydot
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ml",
                      help="Which ML to use tpot / auto-sklearn (default) / sklearn-algo"
                           "default all available sklearn models")
    parser.add_argument("-f", "--filename",
                        help="Filename to process in workloads/ dir: default all swf.gz files")
    parser.add_argument("--droplargeworkloads",
                        help="ignore large workloads")

    args = parser.parse_args()
    #opt = json.dumps(vars(args))
    return args




def get_tt_data(df):
    """
    Returns training and test for given workload
    X_train, X_test are pandas dataframes

    y_* are 1d numpy array
    :param workload:
    :return: X_train, X_test, y_train, y_test
    """



    # Drop fields, cause these are known afterwards
    df = df.drop('CPUTime', axis=1)
    df = df.drop('UsedMEM', axis=1)
    df = df.drop('Status', axis=1)


    # SPlitting date to hour of day ...
    splitdates=1
    if splitdates:
        df = split_dates(df, "SubmitTime")
        df = df.drop('SubmitTime', axis=1)
        # df = df['SubmitTime'].astype('float')
        df = split_dates(df, "StartTime")
        # df = df['StartTime'].astype('float')
        df = df.drop('StartTime', axis=1)

    # dummy = df.to_numpy()

    y = df.pop('RunTime').values
    X_train, X_test, y_train, y_test = train_test_split(df, y)

    return X_train, X_test, y_train, y_test

def workload_set_types(df):
    df['User'] = df['User'].astype('category')
    df['Group'] = df['Group'].astype('category')
    df['Class'] = df['Class'].astype('category')
    df['Partition'] = df['Partition'].astype('category')
    # df['Status'] = df['Status'].astype('category')
    df['Exe'] = df['Exe'].astype('category')

    # Get dtype of column (integer and categorical)
    dtypes = []
    # print(df.dtypes)
    dummy = df.dtypes.tolist()
    df_cols = df.columns.tolist()
    for col in df_cols:
        if df[col].dtype.kind in 'bifc':
            dtypes.append("Numerical")
        else:
            dtypes.append("Categorical")

    return df, dtypes


def main():


    opt = parse_args()



    available_models = ["alea"]

    if opt.droplargeworkloads:
        drop_large_workloads = opt.droplargeworkloads


    workloads = get_all_workloads()
    #removing large workloads
    #Only use on large memory machines
    if drop_large_workloads == True:
        workloads = drop_large_workloads(workloads)
    workloads.sort()
    #create result dir
    resultdir = os.path.join("results/", os.uname()[1], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(resultdir, exist_ok=True)

    workloads = ["fh1_workload.swf.gz",
                "CTC-SP2-1996-3.1-cln.swf.gz",
                 "KTH-SP2-1996-2.1-cln.swf.gz",
                 "SDSC-Par-1996-3.1-cln.swf.gz",
                 "KIT-FH2-2016-1.swf.gz"]

    algo_header = {i: 1 for i in available_models}

    for workload in workloads:
        #read swf file into pandas dataframe
        print(workload)
        df = get_initial_df(workload)
        #df = df.head(1000)
        #df = df.drop('JobId', axis=1)
        df = fix_times(workload, df)

        for algo in available_models:
            algodir = resultdir + "/" + algo


            os.makedirs(algodir,exist_ok=True)


            #use alea preidction technique
            users = df['User'].unique().tolist()
            y_all_predict = []
            y_all_true = []
            y_all_req = []
            pbar = ProgressBar()

            for user in pbar(users):
                #get all jobs for user
                df_user = df.loc[df['User'] == user]
                df_user['ratio'] = df_user['RunTime'].copy() / df_user['ReqWallTime'].copy()
                #df_user['ratio'] = df_user.iloc['RunTime'].div(df_user['ReqWallTime'])

                # initial values for user a.k.a cold start
                ratios = [df_user.iloc[0]['ratio']]
                y_true = df_user['RunTime'].values
                y_predict = [df_user.iloc[0]['ReqWallTime']]
                y_request = df_user['ReqWallTime'].values
                #skip first entry
                for index, row in df_user.iloc[1:].iterrows():
                    avg_ratio = sum(ratios) / len(ratios)
                    refined_time = avg_ratio * row['ReqWallTime']
                    y_predict.append(refined_time)
                    ratios.append(row['ratio'])
                    ratios = ratios[-5:]

                y_all_predict.append(y_predict)
                y_all_req.append(y_request)
                y_all_true.append(y_true)

            y_all_predict = np.concatenate([np.array(i) for i in y_all_predict])
            y_all_req = np.concatenate([np.array(i) for i in y_all_req])
            y_all_true = np.concatenate([np.array(i) for i in y_all_true])

            #prediction and metrics, some dont make sense here
            train_score = "nan"
            test_score = "nan"
            mase = "nan"
            mae = mean_absolute_error(y_all_true, y_all_predict)
            medae = median_absolute_error(y_all_true, y_all_predict)
            r2 = r2_score(y_all_true, y_all_predict)
            mape = mean_absolute_percentage_error(y_all_true, y_all_predict)
            wtld = walltime_lambda_deviation(y_all_true, y_all_predict, y_all_req)

            #Print to stdout
            print("\nTrain score: " + str(train_score))
            print("Test score: " + str(test_score))
            print("MASE: " + str(mase))
            print("MAE: " + str(mae))
            print("MAPE: " + str(mape))
            print("lambda: " + str(wtld))
            print("R2:" + str(r2))
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



            time.sleep(1)
    print("Everything done - HAVE A NICE DAY")


if __name__ == '__main__':
    main()
