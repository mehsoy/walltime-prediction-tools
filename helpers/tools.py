import glob, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_initial_df(workload):
    filename =  "workloads/"  + workload
    #if workload == "SDSC":
    #    filename = "workloads/SDSC.swf"
    #if workload == "CTC":
    #    filename = "workloads/CTC.swf"
    print(filename)
    df = pd.read_csv(filename, compression='gzip',
                     sep="\s+|\t+|\s+\t+|\t+\s+", comment=';',
                     header = None,
                     names=['JobId', "SubmitTime", 'WaitTime', 'RunTime','TaskCount', 'CPUTime',
                            'UsedMEM', 'TaskReq',  'ReqWallTime', 'RequestedMemory',
                            'Status', 'User', 'Group', 'Exe', 'Class', 'Partition',
                            'prejob', 'thinktime'],
                     engine='python')
    #print(df.memory_usage(index=True, deep=False))
    return df

def get_tt_data(df):
    """
    Returns training and test for given workload
    X_train, X_test are pandas dataframes

    y_* are 1d numpy array
    :param workload:
    :return: X_train, X_test, y_train, y_test
    """



    # Drop fields
    df = df.drop('CPUTime', axis=1)
    df = df.drop('UsedMEM', axis=1)
    df = df.drop('Status', axis=1)


    df['User'] = df['User'].astype('category')
    df['Group'] = df['Group'].astype('category')
    df['Class'] = df['Class'].astype('category')
    df['Partition'] = df['Partition'].astype('category')
    # df['Status'] = df['Status'].astype('category')
    df['Exe'] = df['Exe'].astype('category')
    # dummy = df.to_numpy()

    y = df.pop('RunTime').values
    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)

    return X_train, X_test, y_train, y_test

def fix_times(workload,df):
    """
    converting times back to timesteps and adding starttime
    :param  workload: Name of the owrkload to pick right UnixstarTime. Extracted from swf file.
            df: swf loaded into pandas dataframe
    :return: df
    """
    if workload == "ANL-Intrepid-2009-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "CEA-Curie-2011-2.1-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "CIEMAT-Euler-2008-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "CTC-SP2-1995-2.swf.gz":
        UnixStartTime = 1231135224
    if workload == "CTC-SP2-1996-3.1-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "DAS2-fs0-2003-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "DAS2-fs1-2003-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "DAS2-fs2-2003-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "DAS2-fs3-2003-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "DAS2-fs4-2003-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "HPC2N-2002-2.2-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "Intel-NetbatchA-2012-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "Intel-NetbatchB-2012-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "Intel-NetbatchC-2012-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "Intel-NetbatchD-2012-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "KIT-FH2-2016-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "KTH-SP2-1996-2.1-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "LANL-CM5-1994-4.1-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "LANL-O2K-1999-2.swf.gz":
        UnixStartTime = 1231135224
    if workload == "LCG-2005-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "LLNL-Atlas-2006-2.1-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "LLNL-T3D-1996-2.swf.gz":
        UnixStartTime = 1231135224
    if workload == "LLNL-Thunder-2007-1.1-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "LLNL-uBGL-2006-2.swf.gz":
        UnixStartTime = 1231135224
    if workload == "LPC-EGEE-2004-1.2-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "METACENTRUM-2009-2.swf.gz":
        UnixStartTime = 1231135224
    if workload == "METACENTRUM-2013-3.swf.gz":
        UnixStartTime = 1231135224
    if workload == "NASA-iPSC-1993-3.1-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "OSC-Clust-2000-3.1-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "PIK-IPLEX-2009-1.swf.gz":
        UnixStartTime = 1231135224
    if workload == "RICC-2010-2.swf.gz":
        UnixStartTime = 1231135224
    if workload == "Sandia-Ross-2001-1.1-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "SDSC-BLUE-2000-4.2-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "SDSC-DS-2004-2.1-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "SDSC-Par-1995-3.1-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "SDSC-Par-1996-3.1-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "SDSC-SP2-1998-4.2-cln.swf.gz":
        UnixStartTime = 1231135224
    if workload == "SHARCNET-2005-2.swf.gz":
        UnixStartTime = 1231135224
    if workload == "SHARCNET-Whale-2006-2.swf.gz":
        UnixStartTime = 1231135224
    if workload == "UniLu-Gaia-2014-2.swf.gz":
        UnixStartTime = 1231135224
    if workload == "fh1_workload.swf.gz":
        UnixStartTime = 1451615127

    df['SubmitTime'] = df['SubmitTime'] + UnixStartTime
    df['StartTime']  = df['SubmitTime'] + df['WaitTime']
    #df['SubmitTime'] = pd.to_datetime(df['SubmitTime'], unit='s')
    #df['StartTime'] = pd.to_datetime(df['StartTime'], unit='s')
    return df

def drop_large_workloads(workload_list):
    unwanted_workloads = set(['CIEMAT-Euler-2008-1.swf.gz',
                              'Intel-NetbatchA-2012-1.swf.gz',
                              'Intel-NetbatchB-2012-1.swf.gz',
                              'Intel-NetbatchC-2012-1.swf.gz',
                              'Intel-NetbatchD-2012-1.swf.gz',
                              'METACENTRUM-2013-3.swf.gz',
                              'SHARCNET-2005-2.swf.gz'
                              ])
    workloads = [x for x in workload_list if x not in unwanted_workloads]
    return workloads

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true_zeros = np.nonzero(y_true == 0)[0]
    y_pred_zeros = np.nonzero(y_pred == 0)[0]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def MASE(training_series, testing_series, prediction_series):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.

    See "Another look at measures of forecast accuracy", Rob J Hyndman

    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.

    """
    print
    "Needs to be tested."
    n = training_series.shape[0]
    d = np.abs(np.diff(training_series)).sum() / (n - 1)

    errors = np.abs(testing_series - prediction_series)
    return errors.mean() / d

def walltime_lambda_deviation(y_true, y_pred, y_request):
    """
    Computes the lambda deviation

    Mehmet Soysal

    parameters:
        y_true: true values wall clock time a.k.a real run-time of the jobs
        y_pred: predicted wall time values
        y_request: requested wall time by the user (a.k.a useless estimates by the user ;) )
        returns: numpy array lambda values

    """
    walltime_lam = np.abs(y_pred - y_true).mean()  / np.abs(y_request - y_true).mean()


    return walltime_lam

def split_dates(df, column):
    hour_of_day = range(0, 24)
    day_of_week = range(0,7)

    if column == "SubmitTime":
        #df['Day_SubmitTime'] = pd.DatetimeIndex(pd.to_datetime(df[column],unit='s')).day
        #df['day_of_week'] = pd.Series(day_of_week, dtype="category")
        df['SubmitTime_TS'] = df['SubmitTime'].copy()
        df['SubmitTime'] = pd.to_datetime(df['SubmitTime'], unit='s')
        df['Weekday_SubmitTime'] = pd.Categorical(pd.Series(pd.DatetimeIndex(pd.to_datetime(df[column],unit='s')).weekday, dtype="category") , categories=day_of_week)
        df['Hour_SubmitTime'] = pd.Categorical(pd.Series(pd.DatetimeIndex(pd.to_datetime(df[column],unit='s')).hour, dtype="category") , categories=hour_of_day)
    if column == "StartTime":
        df['StartTime_TS'] = df['StartTime'].copy()
        df['StartTime'] = pd.to_datetime(df['StartTime'], unit='s')
        df['Weekday_StartTime'] = pd.Categorical(pd.Series(pd.DatetimeIndex(pd.to_datetime(df[column],unit='s')).weekday, dtype="category") , categories=day_of_week)
        df['Hour_StartTime'] = pd.Categorical(pd.Series(pd.DatetimeIndex(pd.to_datetime(df[column],unit='s')).hour, dtype="category") , categories=hour_of_day)


    return df

def get_all_workloads():
    #os.chdir("workloads/")

    workloads =[]
    for file in glob.glob("workloads/*.gz"):
        workloads.append(file.split("/")[1])
    # workloads = ["ANL-Intrepid-2009-1.swf.gz","CEA-Curie-2011-2.1-cln.swf.gz","CIEMAT-Euler-2008-1.swf.gz",
    #              "CTC-SP2-1995-2.swf.gz","CTC-SP2-1996-3.1-cln.swf.gz","DAS2-fs0-2003-1.swf.gz",
    #              "DAS2-fs1-2003-1.swf.gz","DAS2-fs2-2003-1.swf.gz","DAS2-fs3-2003-1.swf.gz",
    #              "DAS2-fs4-2003-1.swf.gz","HPC2N-2002-2.2-cln.swf.gz","Intel-NetbatchA-2012-1.swf.gz",
    #              "Intel-NetbatchB-2012-1.swf.gz","Intel-NetbatchC-2012-1.swf.gz","Intel-NetbatchD-2012-1.swf.gz",
    #              "KIT-FH2-2016-1.swf.gz","KTH-SP2-1996-2.1-cln.swf.gz","LANL-CM5-1994-4.1-cln.swf.gz",
    #              "LANL-O2K-1999-2.swf.gz","LCG-2005-1.swf.gz","LLNL-Atlas-2006-2.1-cln.swf.gz",
    #              "LLNL-T3D-1996-2.swf.gz","LLNL-Thunder-2007-1.1-cln.swf.gz","LLNL-uBGL-2006-2.swf.gz",
    #              "LPC-EGEE-2004-1.2-cln.swf.gz","METACENTRUM-2009-2.swf.gz","METACENTRUM-2013-3.swf.gz",
    #              "NASA-iPSC-1993-3.1-cln.swf.gz","OSC-Clust-2000-3.1-cln.swf.gz","PIK-IPLEX-2009-1.swf.gz",
    #              "RICC-2010-2.swf.gz","Sandia-Ross-2001-1.1-cln.swf.gz","SDSC-BLUE-2000-4.2-cln.swf.gz",
    #              "SDSC-DS-2004-2.1-cln.swf.gz","SDSC-Par-1995-3.1-cln.swf.gz","SDSC-Par-1996-3.1-cln.swf.gz",
    #              "SDSC-SP2-1998-4.2-cln.swf.gz","SHARCNET-2005-2.swf.gz","SHARCNET-Whale-2006-2.swf.gz",
    #              "UniLu-Gaia-2014-2.swf.gz"]

    return workloads


def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def my_r2_score(y_true, y_pred):
    ssres = np.sum(np.square(y_true - y_pred))
    sstot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - ssres / sstot

