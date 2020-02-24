#!/usr/bin/env python
# Copyright 2019 Markus Götz(markus.goetz@kit.edu)
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
import datetime
import os

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_initial_df(workload):
    filename = os.path.join('workloads', workload)
    df = pd.read_csv(
        filename,
        compression='gzip',
        sep='\s+|\t+|\s+\t+|\t+\s+',
        comment=';',
        header=None,
        names=[
            'JobId', 'SubmitTime', 'WaitTime', 'RunTime', 'TaskCount', 'CPUTime', 'UsedMEM', 'TaskReq', 'ReqWallTime',
            'RequestedMemory', 'Status', 'User', 'Group', 'Exe', 'Class', 'Partition', 'prejob', 'thinktime'
        ],
        engine='python'
    )

    return df


def fix_times(_, df):
    start = 1231135224

    df['SubmitTime'] = df['SubmitTime'] + start
    df['StartTime'] = df['SubmitTime'] + df['WaitTime']

    return df


def split_dates(df, column):
    hour_of_day = range(0, 24)
    day_of_week = range(0, 7)

    df['{}_TS'.format(column)] = df[column].copy()
    df[column] = pd.to_datetime(df[column], unit='s')

    df['Weekday_{}'.format(column)] = pd.Categorical(
        pd.Series(pd.DatetimeIndex(pd.to_datetime(df[column], unit='s')).weekday, dtype='category'),
        categories=day_of_week
    )
    df['Hour_{}'.format(column)] = pd.Categorical(
        pd.Series(pd.DatetimeIndex(pd.to_datetime(df[column], unit='s')).hour, dtype='category'),
        categories=hour_of_day
    )
    df = df.drop(column, axis=1)

    return df


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_scaled_error(training_series, testing_series, prediction_series):
    n = training_series.shape[0]
    d = np.abs(np.diff(training_series)).sum() / (n - 1)

    errors = np.abs(testing_series - prediction_series)
    return errors.mean() / d


def walltime_lambda_deviation(y_true, y_pred, y_request):
    walltime_lam = np.abs(y_pred - y_true).mean() / np.abs(y_request - y_true).mean()

    return walltime_lam


def build_model(data):
    model = Sequential()

    model.add(Dense(40, input_dim=data.shape[-1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    return model


def main():
    workloads = ["fh1_workload.swf.gz",
                 "CTC-SP2-1996-3.1-cln.swf.gz",
                 "KTH-SP2-1996-2.1-cln.swf.gz",
                 "SDSC-Par-1996-3.1-cln.swf.gz",
                 "KIT-FH2-2016-1.swf.gz"]
    workloads.sort()

    # create result dir
    result_dir = os.path.join('results/', os.uname()[1], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(result_dir)

    # iterate each of the workloads
    for workload in workloads:
        # read swf file into pandas dataframe
        print('Loading workload file:' + workload + '\n')
        df = get_initial_df(workload)

        # drop columns with unknown samples
        for column in ['CPUTime', 'UsedMEM', 'Status', 'RequestedMemory', 'Group', 'Exe', 'Partition', 'prejob', 'thinktime']:
            df = df.drop(column, axis=1)

        # fix relative SubmitTime to absolute and add StartTime of job (unix timestamp)
        # split week
        df = fix_times(workload, df)
        df = split_dates(df, 'SubmitTime')
        df = split_dates(df, 'StartTime')

        # one-hot encode categorical and periodic values and concat resulting dataframes
        dataframes = [df]
        for column in ['User', 'Class', 'Weekday_SubmitTime', 'Hour_SubmitTime', 'Weekday_StartTime', 'Hour_StartTime']:
            dataframes.append(pd.get_dummies(df[column].astype(np.int), prefix=column))
            dataframes[0] = dataframes[0].drop(column, axis=1)
        df = pd.concat(dataframes, axis=1)
        del dataframes

        # extract labels
        y = df.pop('RunTime').values
        y = np.reshape(y, (-1, 1))

        # split into train and test set
        x_train, x_test, y_train, y_test = train_test_split(df, y, random_state=42)

        y_request = x_test['ReqWallTime'].values
        y_request = np.reshape(y_request, (-1, 1))

        # convert everything to numpy
        np_x_train = x_train.to_numpy()
        np_y_train = y_train
        np_x_test = x_test.to_numpy()
        np_y_test = y_test

        # standardize data
        mean = np_x_train.mean(axis=0)
        std = np_x_train.std(axis=0)
        std[std == 0] = 1

        np_x_train = (np_x_train - mean) / std
        np_x_test = (np_x_test - mean) / std

        # remove unneeded stuff
        del [[df, x_test, x_train, y_train, y_test]]

        # train the model
        regressor = build_model(np_x_train)
        regressor.fit(np_x_train, np_y_train, epochs=20, batch_size=8, verbose=1, validation_split=0.2, callbacks=[ModelCheckpoint('{}/model.h5'.format(result_dir))])

        # perform a prediction
        y_predict = regressor.predict(np_x_test)
        print(y_predict)

        # compute metrics
        train_loss = float(regressor.evaluate(np_x_train, np_y_train)[0])
        test_loss = float(regressor.evaluate(np_x_test, np_y_test)[0])
        mae = mean_absolute_error(np_y_test, y_predict)
        mape = mean_absolute_percentage_error(np_y_test, y_predict)
        mase = mean_absolute_scaled_error(np_y_train, np_y_test, y_predict)
        meae = median_absolute_error(np_y_test, y_predict)
        r2 = r2_score(np_y_test, y_predict)
        wtld = walltime_lambda_deviation(np_y_test, y_predict, y_request)

        # print the metrics
        print('\n')
        print('Train loss:', train_loss)
        print('Test loss:', test_loss)
        print('MAE:', mae)
        print('MAPE:', mape)
        print('MASE:', mase)
        print('MEAE:', meae)
        print('R2:', r2)
        # print('Lambda:', wtld)

        # save results to disk
        model_kind = 'keras'
        result_file = '{}/result-{}.txt'.format(result_dir, model_kind)
        with open(result_file, 'a') as output_handle:
            output_handle.write('{}:{}:{}:{}:{}:{}:{}:{}:{}:{}\n'.format(
                workload, model_kind, train_loss, test_loss, mae, mape, mase, meae, r2, wtld
            ))


if __name__ == '__main__':
    main()
