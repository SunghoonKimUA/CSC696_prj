# starts from https://keras.io/examples/timeseries/timeseries_classification_from_scratch/ and https://keras.io/examples/generative/lstm_character_level_text_generation/
import numpy as np
import tensorflow as tf
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

from ds_lib import *

# var1_idx : ev vs ts in directory
# var2_idx : ex vs nm in directory
# var3_idx : 0~20 in directory
# file_idx : file_name
# ctry_cd : country code
# read_mode : read ctry_1 only vs read both ctry_1 and ctry_2

# 3. design model
def make_model(input_shape1, archi):
    hidden_units = 32

    if archi == 'rnn':
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape1),
                layers.SimpleRNN(hidden_units),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif archi == 'lstm':
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape1),
                layers.LSTM(hidden_units),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    elif archi == 'gru':
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape1),
                layers.GRU(hidden_units),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    """
    model = keras.Sequential(
        [
            layers.Reshape(new_shape, input_shape=input_shape1),
            keras.Input(shape=new_shape),
            layers.GRU(hidden_units),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    """

    return model

def training_model(X_train_arr, y_train, X_test_arr, y_test, idx, archi):
    model = make_model(input_shape1=X_train_arr.shape[1:], archi=archi)
    # model.summary()

    # 4. train the model
    epochs = 100
    batch_size = 128

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "./model/best_timeseries1_cnt_"+idx+"_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=0),
    ]
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy","mse"],
    )

    history = model.fit(
        X_train_arr,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=0,
    )

    # 5. evaluate
    #    (prev_grade = curr_grade vs model's prediction) by accuracy
    model = keras.models.load_model("./model/best_timeseries1_cnt_"+idx+"_model.h5")

    return model.evaluate(X_test_arr, y_test)

print("Start of the experiment", datetime.now().strftime("%H:%M:%S"))

performance_dist = []
performance_epochs = 3

import itertools
from more_itertools import random_product

option_lists = [
#   ['ev', 'ts'],
   ['ev'],
   ['ex', 'nm'],#
   ['0', '5', '10', '15', '20', '30'],#date gap between news and eval'0', '5', '10', '15', '20', '30'
   ['1', '2'],#read_mode(subj vs subj+obj), '2'
   [5, 10],# number of event_code, 5, 10, 15
   [0, 1],#include economic indicators or not, 0, 1
]

# for arr in experiment_arr:
#    print(arr)

date_range_lst = pd.read_csv("./data/cre-crc-historical-internet-english.csv", index_col=0).columns[2:]

S_indic = pd.read_csv("./data/imf_df.csv", index_col=0, low_memory=False)

# 0. normalizing economic_indicator data
S_indic = S_indic.replace(99999,0)
normalized_indic=(S_indic-S_indic.mean())/S_indic.std()
normalized_indic['ctry_cd'] = S_indic['ctry_cd']
normalized_indic['dt_dtr']  = S_indic['dt_dtr']
# print(normalized_indic)
# print(get_indicator_score(normalized_indic, '22-Jan-99', 'USA'))

timeseries_cand = [60, 36, 12, 6, 3]#60, 36, 12, 6, 3
archi_cand = ['rnn', 'lstm', 'gru']#
norm_prd_cnt = 10

for history_len in timeseries_cand:
    for p_idx in range(performance_epochs):
        S = pd.read_csv("./data/train_timeseries_"+str(norm_prd_cnt)+"_normed_"+str(history_len)+"_evaldt_df.csv", index_col=0)

        # experiment_arr = itertools.product(*option_lists)
        epochs_ = 5
        tmp_tuples = random_product(*option_lists, repeat=epochs_)
        tup_cnt = int(len(tmp_tuples)/epochs_)
        experiment_arr = np.ndarray(shape=(epochs_), dtype=object)
        for ep_idx in range(epochs_):
            experiment_arr[ep_idx] = tmp_tuples[tup_cnt*ep_idx:tup_cnt*(ep_idx+1)]
        # print(experiment_arr)

        # 1. shuffle datasets
        S = shuffle(S)
        # print(S.shape)
        print("Start Time of iteration("+str(p_idx)+"/"+str(history_len)+") =", datetime.now().strftime("%H:%M:%S"))

        # 1-1. prepare y set
        grd_col_list = [col for col in S.columns if col.startswith('dt_')]
        S_grades = S[grd_col_list]
        S_grades.reset_index(inplace=True, drop=True)
        # target becomes [-1, 0, 1]
        y = S_grades.to_numpy()[:,-1:] - S_grades.to_numpy()[:,-2:-1]
        y = np.sign(y)

        num_classes = len(np.unique(y))
        from tensorflow.keras.utils import to_categorical
        y = to_categorical(y, num_classes=num_classes)
        # print(y.shape)

        # 2. declare train_sets for only 60 of grades history and split train vs test (7 vs 3)
        X_hist_only = S_grades.to_numpy()[:,:-1]
        # print(X_hist_only.shape)

        X_train, X_test, y_train, y_test = train_test_split(X_hist_only, y, test_size=0.3)
        # 2-1. build base_line which replicates last_grades into y_hat
        X_test_baseline = np.zeros(shape=(y_test.shape[0],1))
        y_test_baseline = np.reshape([np.argmax(i, axis=None, out=None) for i in y_test], (y_test.shape[0],1))

        # 2-2. estimate base_line's accuracy
        m = tf.keras.metrics.Accuracy()
        m.update_state(X_test_baseline, y_test_baseline)
        base_acc = m.result().numpy()

        m = tf.keras.metrics.MeanSquaredError()
        m.update_state(X_test_baseline, y_test_baseline)
        base_mse = m.result().numpy()

        print("★★★Base Line's Accuracy:",base_acc)
        idx = 'BaseLine'
        performance_dist.append([idx, history_len, '', p_idx, 0.0, base_acc, base_mse])

        X_train_arr = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_arr  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        idx = 'grades_only'
        test_loss, test_acc, test_mse = training_model(X_train_arr, y_train, X_test_arr, y_test, idx, 'gru')
        print("★★★Grades only's Accuracy:",test_acc)

        performance_dist.append([idx, history_len, 'gru', p_idx, test_loss, test_acc, test_mse])

        # 3. prepare X set, which contain country_cd, 60 grades, and 60 eval_dt
        org_X = S[[col for col in S.columns if not col.endswith(str(history_len))]].to_numpy()
        # print(org_X.shape) # (3459, 121)

        for var1_idx, var2_idx, var3_idx, read_mode, num_of_evnt_cd, bool_indicators in experiment_arr:
            # 3-0. open summary csv files
            Sum, Sum1 = open_summary_csv(var1_idx, var2_idx, var3_idx, '_cnt')
            gdelt_idx = var1_idx+"_"+var2_idx+"_"+var3_idx
            m_feat_1 = get_major_features(var1_idx, var2_idx, var3_idx, 1, num_of_evnt_cd)
            m_feat_2 = get_major_features(var1_idx, var2_idx, var3_idx, 2, num_of_evnt_cd)
            m_up_feat_1, m_down_feat_1 = get_major_features_sep(var1_idx, var2_idx, var3_idx, 1, num_of_evnt_cd)
            m_up_feat_2, m_down_feat_2 = get_major_features_sep(var1_idx, var2_idx, var3_idx, 2, num_of_evnt_cd)
            # print(gdelt_idx, m_up_feat_1, m_down_feat_1)

            # 3-1. Estimate new axis's length, grade(1) + new_avg_score(if read_mode == 1 then 1 else 2)
            news_arr_len = 1+2
            if read_mode == '2':
                news_arr_len = news_arr_len + 2
            if bool_indicators == 1:
                news_arr_len = news_arr_len + 1
            #### to normalize "news score", it needs at least norm_trgt_cnt timeslots, therefore trial datasets become shrinked
            X = np.ndarray(shape=(org_X.shape[0], history_len, news_arr_len))

            # 3-2. build X set, which contains 60 grades and gdelt_arr of each grade
            """
            print(gdelt_idx+"//"+str(get_normzed_avg_gdelt_score(Sum, m_up_feat_1, 86, 'RUS', gdelt_idx+"_up", '1', norm_prd_cnt)))
            print(gdelt_idx+"//"+str(get_normzed_avg_gdelt_score(Sum, m_up_feat_1, 87, 'RUS', gdelt_idx+"_up", '1', norm_prd_cnt)))
            print(gdelt_idx+"//"+str(get_normzed_avg_gdelt_score(Sum, m_up_feat_1, 88, 'RUS', gdelt_idx+"_up", '1', norm_prd_cnt)))
            print(gdelt_idx+"//"+str(get_normzed_avg_gdelt_score(Sum, m_down_feat_1, 86, 'RUS', gdelt_idx+"_down", '1', norm_prd_cnt)))
            print(gdelt_idx+"//"+str(get_normzed_avg_gdelt_score(Sum, m_down_feat_1, 87, 'RUS', gdelt_idx+"_down", '1', norm_prd_cnt)))
            print(gdelt_idx+"//"+str(get_normzed_avg_gdelt_score(Sum, m_down_feat_1, 88, 'RUS', gdelt_idx+"_down", '1', norm_prd_cnt)))
            """
            for dim1_idx in range(org_X.shape[0]):
                # print(str(dim1_idx)+"//"+str(org_X.shape[0]))
                for dim2_idx in range(history_len):
                    # 1st column : grade
                    col_idx = 0
                    X[dim1_idx,dim2_idx,col_idx] = org_X[dim1_idx, dim2_idx+1]
                    col_idx = col_idx + 1
                    # 2nd, 3rd column : get_avg_gdelt_score in up_features/down_features which read_mode = 1
                    X[dim1_idx,dim2_idx,col_idx] = max(0.0,get_normzed_avg_gdelt_score(Sum, m_up_feat_1, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+1]), org_X[dim1_idx, 0], gdelt_idx+"_up", '1', norm_prd_cnt))
                    col_idx = col_idx + 1
                    X[dim1_idx,dim2_idx,col_idx] = min(0.0,get_normzed_avg_gdelt_score(Sum, m_down_feat_1, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+1]), org_X[dim1_idx, 0], gdelt_idx+"_down", '1', norm_prd_cnt))
                    col_idx = col_idx + 1
                    # next column : get_avg_gdelt_score which read_mode = 2
                    if read_mode == '2':
                        X[dim1_idx,dim2_idx,col_idx] = max(0.0,get_normzed_avg_gdelt_score(Sum1, m_up_feat_2, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+1]), org_X[dim1_idx, 0], gdelt_idx+"_up", '2', norm_prd_cnt))
                        col_idx = col_idx + 1
                        X[dim1_idx,dim2_idx,col_idx] = min(0.0,get_normzed_avg_gdelt_score(Sum1, m_down_feat_2, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+1]), org_X[dim1_idx, 0], gdelt_idx+"_down", '2', norm_prd_cnt))
                        col_idx = col_idx + 1
                    # next column : get_indicator_score
                    if bool_indicators == 1:
                        X[dim1_idx,dim2_idx,col_idx] = get_indicator_score(normalized_indic, org_X[dim1_idx, dim2_idx+history_len+1], org_X[dim1_idx, 0])
                        col_idx = col_idx + 1

            # print(X)
            print("After build X("+str(p_idx)+"_"+build_idx(var1_idx, var2_idx, var3_idx, read_mode, num_of_evnt_cd, bool_indicators)+") =", datetime.now().strftime("%H:%M:%S"))

            # 3-3. split train vs test (7 vs 3) and train model and normalizing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            # X_train_arr = X_train.reshape((X_train.shape[0], X_train.shape[1], news_arr_len))
            # X_test_arr  = X_test.reshape((X_test.shape[0], X_test.shape[1], news_arr_len))

            for archi in archi_cand:
                idx = build_idx(var1_idx, var2_idx, var3_idx, read_mode, num_of_evnt_cd, bool_indicators)+"_"+archi+"__"+str(history_len)
                test_loss, test_acc, test_mse = training_model(X_train, y_train, X_test, y_test, idx, archi)
                performance_dist.append([idx, history_len, archi, p_idx, test_loss, test_acc, test_mse])

            print("★★★Model's accuracy", test_acc)
            print("After training X("+str(p_idx)+"_"+build_idx(var1_idx, var2_idx, var3_idx, read_mode, num_of_evnt_cd, bool_indicators)+") =", datetime.now().strftime("%H:%M:%S"))

pd.DataFrame(performance_dist, columns=['input_option', 'history_len', 'archi', 'iter_no', 'loss', 'accuracy', 'mean squared error']).to_csv("./data/choose_his_len_from_normzed_cnt.csv")
print("End of the experiment", datetime.now().strftime("%H:%M:%S"))
