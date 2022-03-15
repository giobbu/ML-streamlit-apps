
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from util import logging
import tensorflow as tf
import streamlit as st



def data_reader(path_file):
    data = pd.read_csv(path_file)
    nRow, nCol = data.shape
    logging.info("-- read data")
    return data


def streets_selection(THRESHOLD, DATAFRAME):

    DATAFRAME_STREETS = DATAFRAME.iloc[:,1:] # without index
    LST_STREETS = list(DATAFRAME_STREETS.columns.values)

    LST_MEAN_VALUE = []
    LST_STREETS_NEW = []

    for street in LST_STREETS:
        values_street = DATAFRAME_STREETS[street]
        mean_value_street = np.mean(values_street)
        LST_MEAN_VALUE.append(mean_value_street)
        LST_STREETS_NEW.append(street)

    # create dataframe with street index column and avg value of flow/vel
    DF_MEAN = pd.DataFrame({'street_index': LST_STREETS_NEW, 'mean_value': LST_MEAN_VALUE})
    # select streets with avg value higher than threshold
    SLCT_STREETS = DF_MEAN[(DF_MEAN ['mean_value']>= THRESHOLD)] 
    SLCT_STREETS = SLCT_STREETS.sort_values(by=['street_index'])
    LST_SLCT_STREETS = list(SLCT_STREETS.street_index)

    logging.info("-- select streets with thresh: "+str(THRESHOLD))
    return LST_SLCT_STREETS



def feat_engin(MEAN_VALUE, dataframe):

    # select streets with avg flow/velocity higher than threshold "MEAN_VALUE"
    LST_SLCT_STREETS = streets_selection(MEAN_VALUE, dataframe)

    # convert timestamp into datetime column
    dataframe['Datetime'] = pd.to_datetime(dataframe['datetime'])
    # copy dataframe
    DATAFRAME_ = dataframe
    DATAFRAME_ = DATAFRAME_.drop(['datetime'],axis=1) 
    DATAFRAME_ = DATAFRAME_[DATAFRAME_.columns.intersection(LST_SLCT_STREETS)]

    
    # Time-based Covariates
    # minutes, hours
    DATAFRAME_['minutes'] = dataframe['Datetime'].dt.minute
    DATAFRAME_['hour'] = dataframe['Datetime'].dt.hour
    # sin cos of hours
    DATAFRAME_['hour_x'] = np.sin(DATAFRAME_.hour*(2.*np.pi/23))
    DATAFRAME_['hour_y'] = np.cos(DATAFRAME_.hour*(2.*np.pi/23))
    # day, dayofweek
    DATAFRAME_['day'] = dataframe['Datetime'].dt.day
    DATAFRAME_['DayOfWeek'] = dataframe['Datetime'].dt.dayofweek
    # encode workindays, weekend days
    DATAFRAME_['WorkingDays'] = DATAFRAME_['DayOfWeek'].apply(lambda y: 2 if y < 5 else y)
    DATAFRAME_['WorkingDays'] = DATAFRAME_['WorkingDays'].apply(lambda y: 1 if y == 5 else y)
    DATAFRAME_['WorkingDays'] = DATAFRAME_['WorkingDays'].apply(lambda y: 0 if y == 6 else y)
    # dump it
    DATAFRAME_ = DATAFRAME_.drop(['minutes','hour','day'],axis=1)

    timestamp = dataframe['datetime']
    logging.info("-- add time-based features")
    return DATAFRAME_, LST_SLCT_STREETS, timestamp



def data_split_scale(dataframe, VALID_LENGTH, TEST_LENGTH, N_FEAT_TIME, TIMESTAMP):
    
    # define scaler
    scaler = MinMaxScaler(feature_range=(0, 1)) # for streets
    scaler_aux = MinMaxScaler(feature_range=(0, 1)) # for time-based ft

    # datframe without time-based features
    df_no_time = dataframe.values[:, : -N_FEAT_TIME]
    # split train/val/test
    train_no_time = df_no_time[ : -VALID_LENGTH -TEST_LENGTH]
    valid_no_time = df_no_time[ -VALID_LENGTH -TEST_LENGTH: -TEST_LENGTH]
    test_no_time = df_no_time[ -TEST_LENGTH : ]
    # scale the data
    sc_train_no_time = scaler.fit_transform(train_no_time) # fit & transform
    sc_valid_no_time = scaler.transform(valid_no_time) # transform (not known in advance)
    sc_test_no_time = scaler.transform(test_no_time) # transform (not known in advance)

    # datframe of only time-based features
    df_time = dataframe.values[:, -N_FEAT_TIME:]
    # time-based features are known in advance
    sc_df_time = scaler_aux.fit_transform(df_time)
    # split train/valid/test
    sc_train_time = sc_df_time[ : -VALID_LENGTH -TEST_LENGTH]
    sc_valid_time = sc_df_time[ -VALID_LENGTH -TEST_LENGTH : -TEST_LENGTH]
    sc_test_time= sc_df_time[ -TEST_LENGTH : ]

    # concatenate dataframes streets and time-based features
    train_set = np.hstack([sc_train_no_time, sc_train_time])
    valid_set = np.hstack([sc_valid_no_time, sc_valid_time])
    test_set = np.hstack([sc_test_no_time, sc_test_time])

    timestamp_test = TIMESTAMP[ -TEST_LENGTH : ]
    logging.info("-- scale and split the data:")
    logging.info("         training: "+str(train_set.shape[0])+ ", validation: "+str(valid_set.shape[0])+ ", testing: "+str(test_set.shape[0]))
    return train_no_time, valid_no_time, test_no_time, train_set, valid_set, test_set, scaler, timestamp_test 


def sequence_in_out_naive(dataframe, INPUT, OUTPUT ):
    X, y = list(), list()
    for i in range(len(dataframe)):
        # find the end of this pattern
        end_ix = i + INPUT
        out_end_ix = end_ix + OUTPUT
        # check if we are beyond the dataset
        if out_end_ix > len(dataframe):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = dataframe[i:end_ix, :], dataframe[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def data_loader_naive(dataframe, INPUT, OUTPUT, VALID_LENGTH, TEST_LENGTH):

    X, Y =  sequence_in_out_naive(dataframe, INPUT, OUTPUT )
    X1, Y1 = X[: -VALID_LENGTH -TEST_LENGTH], Y[: -VALID_LENGTH -TEST_LENGTH]
    X2, Y2 = X[ : -TEST_LENGTH +23], Y[ : -TEST_LENGTH +23]
    X3, Y3 = X[-TEST_LENGTH +23 : ], Y[-TEST_LENGTH +23 : ]

    return X1, Y1, X2, Y2, X3, Y3



def data_loader(dataframe, INPUT, OUTPUT, AUX, BATCH):

    TOTAL = INPUT + OUTPUT
    dataset_feat = tf.data.Dataset.from_tensor_slices(dataframe)    
    aux = tf.data.Dataset.from_tensor_slices(dataframe[:, - AUX:])    
    dataset_labels = tf.data.Dataset.from_tensor_slices(dataframe)

    # features - past observations
    feat = dataset_feat.window(INPUT,  shift=1,  stride=1,  drop_remainder=True) 
    feat = feat.flat_map(lambda window: window.batch(INPUT))
    
    # aux - temporal features
    aux = aux.window(OUTPUT,  shift=1,  stride=1,  drop_remainder=True ).skip(INPUT)
    aux = aux.flat_map(lambda window: window.batch(OUTPUT))
    
    # labels - future observations
    label = dataset_labels.window(OUTPUT, shift=1,  stride=1,  drop_remainder=True).skip(INPUT)
    label = label.flat_map(lambda window: window.batch(OUTPUT))
    
    dataset = tf.data.Dataset.zip(((feat, aux), label))

    dataset = dataset.batch(BATCH).prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

