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
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc

import os,datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", default="chosen",
                        help="Filename to process in workloads/* dir: all / chosen / <filename>")
    parser.add_argument("--droplargeworkloads", action='store_true',
                        help="ignore large workloads")
    parser.add_argument("--cpu", action='store_true',
                        help="use only cpu")

    args = parser.parse_args()
    #opt = json.dumps(vars(args))
    return args


def build_regressor(shape):
    model = Sequential()
    model.add(Dense(20, input_dim=shape, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer="adam", metrics=['mae'])
    #model.compile(loss='mse', optimizer="adam", metrics=['mae'])


    return model

def main():


    opt = parse_args()

    if opt.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(tf.test.gpu_device_name())
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Using only CPU version of TF")
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


    available_models = ["keras"]
    algo_header = {i: 1 for i in available_models}
    for workload in workloads:
        #create result dir

        algodir = resultdir + "/keras"


        os.makedirs(algodir, exist_ok=True)


        #read swf file into pandas dataframe
        print("Loading workload file:" + workload + "\n")
        df_dummy = get_initial_df(workload)
        #df =  df_dummy.head(1000).copy()
        df = df_dummy.copy()
        del [[df_dummy]]
        gc.collect()
        #Drop fields cause these values are not know  for new sample
        df = df.drop('CPUTime',axis=1)
        df = df.drop('UsedMEM', axis=1)
        df = df.drop('Status', axis=1)
        #df = df.drop('JobId', axis=1)

        #Fix relative Suibmittime to absolute
        #and add Starttime of job (unixtimestamp)

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


        #get maximum walltime
        max_req_walltime = df['ReqWallTime'].max().copy()
        max_used_walltime = df['RunTime'].max().copy()
        max_walltime = 0


        y = df.pop('RunTime').values
        y = np.reshape(y, (-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)
        y_request = X_test['ReqWallTime'].values
        y_request = np.reshape(y_request, (-1, 1))
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        # already numpy, just for convience
        np_x_train = X_train
        np_x_test =  X_test

        np_y_train = y_train
        np_y_test = y_test

        #remove unneeded stuff
        del [[df,X_test,X_train,y_train,y_test]]
        gc.collect()

        use_model = "keras"
        #regressor = KerasRegressor(build_fn=build_regressor)
        regressor = build_regressor(np_x_train.shape[-1])
        regressor.fit(np_x_train, np_y_train,
                                epochs=3, batch_size=64,
                                verbose=1, validation_split=0.3)

        y_predict = regressor.predict(np_x_test)

        #reduce accuracy du to low memory machines
        #y_predict = y_predict.astype('int16')
        #y_request = y_request.astype('int16')


        #train_score = float(regressor.score(np_x_train, np_y_train))
        #test_score = float(regressor.score(np_x_test, np_y_test))
        train_score = float(regressor.evaluate(np_x_train, np_y_train)[0])
        test_score = float(regressor.evaluate(np_x_test, np_y_test)[0])

        mae = mean_absolute_error(np_y_test, y_predict)
        medae = median_absolute_error(np_y_test, y_predict)
        mase = MASE(np_y_train, np_y_test, y_predict)
        r2 = r2_score(np_y_test, y_predict)
        mape = mean_absolute_percentage_error(np_y_test, y_predict)

        #issue with array allocation
        wtld = walltime_lambda_deviation(np_y_test, y_predict, y_request)
        # print the metrics
        print('\n')
        print('Train loss:', train_score)
        print('Test loss:', test_score)
        print('MAE:', mae)
        print('MAPE:', mape)
        print('MASE:', mase)
        print('MEAE:', medae)
        print('R2:', r2)
        print('Lambda:', wtld)

        # save results to disk
        algo = 'keras'
        result_file = '{}/result-{}.txt'.format(resultdir, use_model)
        with open(result_file, 'a') as output_handle:
            if algo_header[algo] == 1:
                output_handle.write("#Workload;algorithm;train_score;test_score;mae;mape;mase;medae;r2;wltd")
                output_handle.write("\n")
                algo_header[algo] = 0
            output_handle.write('{}:{}:{}:{}:{}:{}:{}:{}:{}:{}\n'.format(
                workload, use_model, train_score, test_score, mae, mape, mase, medae, r2
            ))
        #if use_model == "tpot":
        #    model.export(resultfile + ".tpot.py")
        # time.sleep(20)
        time.sleep(1)
if __name__ == '__main__':
    main()
