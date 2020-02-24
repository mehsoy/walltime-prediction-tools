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
import argparse
import os
import time
from helpers.tools import *
# from helpers.tools import get_initial_df, get_all_workloads, fix_times, mean_absolute_percentage_error
# from helpers.tools import MASE, walltime_lambda_deviation, split_dates, get_tt_data, drop_large_workloads
import pandas as pd
import sklearn
import tensorflow as tf
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dropout, Dense
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler, MaxAbsScaler

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, median_absolute_error
from sklearn.model_selection import train_test_split

from hyperas import optim
from hyperas.distributions import choice, uniform



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc

import os,datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", default="chosen",
                        help="Filename to process in workloads/* dir: all / chosen / <filename>")
    parser.add_argument("--droplargeworkloads", action='store_true',
                        help="ignore large workloads")

    args = parser.parse_args()
    #opt = json.dumps(vars(args))
    return args

def model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense({{choice([8, 16, 32])}}, input_shape=(19,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([8, 16, 32])}}))
    model.add(Activation({{choice(['relu', 'linear'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer="adam", metrics=['mae'])

    result = model.fit(x_train, y_train,
              batch_size={{choice([4,8,16,32, 64, 128])}},
              epochs=50,
              verbose=1,
               validation_data=(x_test, y_test))

    #get the highest validation accuracy of the training epochs
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def get_all_workloads():
    import glob, os
    #os.chdir("workloads/")

    workloads =[]
    for file in glob.glob("workloads/*.gz"):
        workloads.append(file.split("/")[1])

    return workloads


def data():
    workload = "KIT-FH2-2016-1.swf.gz"
    df_dummy = get_initial_df(workload)
    df = df_dummy.copy()
    del [[df_dummy]]
    gc.collect()
    # Drop fields cause these values are not know  for new sample
    df = df.drop('CPUTime', axis=1)
    df = df.drop('UsedMEM', axis=1)
    df = df.drop('Status', axis=1)
    # df = df.drop('JobId', axis=1)

    # Fix relative Suibmittime to absolute
    # and add Starttime of job (unixtimestamp)

    # Fix relative Suibmittime to absolute
    # and add Starttime of job (unixtimestamp)
    df = fix_times(workload, df)
    df = split_dates(df, "SubmitTime")
    df = split_dates(df, "StartTime")
    # Where are dropping original times
    # df contains now *_TS as numric timestamp fileds
    df = df.drop('StartTime', axis=1)
    df = df.drop('SubmitTime', axis=1)
    # change type of some columns / usefull for sklearn
    df['User'] = df['User'].astype('category')
    df['Group'] = df['Group'].astype('category')
    df['Class'] = df['Class'].astype('category')
    df['Partition'] = df['Partition'].astype('category')
    df['Exe'] = df['Exe'].astype('category')
    # dummy = df.to_numpy()

    # get maximum walltime
    max_req_walltime = df['ReqWallTime'].max().copy()
    max_used_walltime = df['RunTime'].max().copy()
    max_walltime = 0

    # linear_field = ['JobId',  'WaitTime', 'TaskCount',
    #                     'TaskReq',  'ReqWallTime', 'RequestedMemory',
    #                      'prejob', 'thinktime', 'StartTime_TS', 'SubmitTime_TS']
    # categories = ['User', 'Group', 'Exe', 'Class', 'Partition', 'Weekday_SubmitTime',
    #               'Hour_SubmitTime','Weekday_StartTime','Hour_StartTime']
    #
    # df_linear = df[linear_field].copy()
    # data = df_linear.to_numpy()
    # for cat in categories:
    #     zipBinarizer = LabelBinarizer().fit(df[cat])
    #     dummy = zipBinarizer.transform(df[cat])
    #
    #     #dummy = df[cat].to_numpy()
    #     #dummy2 = to_categorical(dummy)
    #     data = np.hstack([data, dummy])
    #
    # if max_used_walltime > max_req_walltime:
    #     max_walltime = max_used_walltime + 1
    # else:
    #     max_walltime = max_req_walltime + 1
    y = df.pop('RunTime').values
    y = np.reshape(y, (-1, 1))
    # x = df.to_numpy()
    # scaler_x = MinMaxScaler()
    # scaler_y = MinMaxScaler()
    # print(scaler_x.fit(x))
    # xscale = scaler_x.transform(x)
    # print(scaler_y.fit(y))
    # yscale = scaler_y.transform(y)
    # X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

    d_X_train, d_X_test, d_y_train, d_y_test = train_test_split(df, y, random_state=42)
    y_request = d_X_test['ReqWallTime'].values

    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)
    y_request = X_test['ReqWallTime'].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # convert everything to numpy
    x_train = X_train
    x_test = X_test
    # already numpy, just for convience
    y_train = y_train
    y_test = y_test
    # remove unneeded stuff
    del [[df, d_X_test, d_X_train, d_y_train, d_y_test]]
    gc.collect()
    return x_train, y_train,  x_test, y_test

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



    for workload in workloads:
        #create result dir

        algodir = resultdir + "/keras"


        os.makedirs(algodir, exist_ok=True)


        #read swf file into pandas dataframe
        print("Loading workload file:" + workload + "\n")
        df_dummy = get_initial_df(workload)
        #df =  df_dummy.head(10000).copy()

        x_train, y_train, x_test, y_test = data()
        best_run, best_model = optim.minimize(model=model,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=5,
                                              trials=Trials())
        print("Evalutation of best performing model:")
        print(best_model.evaluate(x_test, y_test))
        print("Best performing model chosen hyper-parameters:")
        print(best_run)



        # history = regressor.fit(np_x_train, np_y_train,
        #                         epochs=4, batch_size=129,
        #                         verbose=1, validation_split=0.2)
        # print(history.history.keys())
        # #                    batch_size=batch_size,
        # #                    epochs=epochs,
        # #                    verbose=1,
        # #                    validation_data=(np_x_test, np_y_test))
        # #y_pred= regressor.predict(np_x_test)
        # y_predict = regressor.predict(np_x_test)
        # train_score = float(regressor.score(np_x_train, np_y_train))
        # test_score = float(regressor.score(np_x_test, np_y_test))
        # mae = mean_absolute_error(np_y_test, y_predict)
        # medae = median_absolute_error(np_y_test, y_predict)
        # mase = MASE(np_y_train, np_y_test, y_predict)
        # r2 = r2_score(np_y_test, y_predict)
        # mape = mean_absolute_percentage_error(np_y_test, y_predict)
        #
        # wtld = walltime_lambda_deviation(np_y_test, y_predict, y_request)
        # print("\nTrain score: " + str(train_score))
        # print("Test score: " + str(test_score))
        # print("MASE: " + str(mase))
        # print("MAE: " + str(mae))
        # print("MAPE: " + str(mape))
        # print("lambda: " + str(wtld))
        # print("MedAE: " + str(medae))
        # resultfile = resultdir + "/keras/result-keras.txt"
        # f = open(resultfile, "a")
        # f.write(str(workload) + ":" + use_model + ":" + str(train_score) +
        #         ":" + str(test_score) +
        #         ":" + str(mase) +
        #         ":" + str(mae) +
        #         ":" + str(medae) +
        #         ":" + str(mape) +
        #         ":" + str(wtld) +
        #         ":" + str(r2))
        # f.write("\n")
        # f.close
        # #if use_model == "tpot":
        # #    model.export(resultfile + ".tpot.py")
        # # time.sleep(20)
        # time.sleep(1)
if __name__ == '__main__':
    main()
