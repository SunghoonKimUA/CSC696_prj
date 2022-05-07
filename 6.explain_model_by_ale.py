# Explainability with ALE for model itself
import pandas as pd
import matplotlib.pyplot as plt
from alibi.explainers import ALE, plot_ale
from sklearn.utils import shuffle

from tensorflow import keras

import numpy as np
from ds_lib import *

var1_idx = 'ev'
var2_idx = 'nm'
var3_idx = '20'
read_mode = '1'
num_of_evnt_cd = 5
bool_indicators = 1
history_len = 36
idx = build_idx(var1_idx, var2_idx, var3_idx, read_mode, num_of_evnt_cd, bool_indicators)+"_tone__"+str(history_len)
norm_prd_cnt = 10

def build_dataset():
    S_indic = pd.read_csv("./data/imf_df.csv", index_col=0, low_memory=False)

    # 0. normalizing economic_indicator data
    S_indic = S_indic.replace(99999,0)
    normalized_indic=(S_indic-S_indic.mean())/S_indic.std()
    normalized_indic['ctry_cd'] = S_indic['ctry_cd']
    normalized_indic['dt_dtr']  = S_indic['dt_dtr']

    # S = pd.read_csv("./data/train_timeseries_"+str(history_len)+"_evaldt_df.csv", index_col=0)
    S = pd.read_csv("./data/train_timeseries_10_normed_"+str(history_len)+"_evaldt_df.csv", index_col=0)
    S = shuffle(S)
    date_range_lst = pd.read_csv("./data/cre-crc-historical-internet-english.csv", index_col=0).columns[2:]

    org_X = S[[col for col in S.columns if not col.endswith(str(history_len))]].to_numpy()

    # Sum, Sum1 = open_summary_csv(var1_idx, var2_idx, var3_idx)
    Sum, Sum1 = open_summary_csv(var1_idx, var2_idx, var3_idx)
    gdelt_idx = var1_idx+"_"+var2_idx+"_"+var3_idx
    m_feat_1 = get_major_features(var1_idx, var2_idx, var3_idx, 1, num_of_evnt_cd)
    m_feat_2 = get_major_features(var1_idx, var2_idx, var3_idx, 2, num_of_evnt_cd)
    m_up_feat_1, m_down_feat_1 = get_major_features_sep(var1_idx, var2_idx, var3_idx, 1, num_of_evnt_cd)
    m_up_feat_2, m_down_feat_2 = get_major_features_sep(var1_idx, var2_idx, var3_idx, 2, num_of_evnt_cd)
    # print(m_down_feat_1)

    # 3-1. Estimate new axis's length, grade(1) + new_avg_score(if read_mode == 1 then 1 else 2)
    news_arr_len = 1+2
    if read_mode == '2':
        news_arr_len = news_arr_len + 2
    if bool_indicators == 1:
        news_arr_len = news_arr_len + 1
    X = np.ndarray(shape=(int(org_X.shape[0]/1), history_len*news_arr_len))
    feat_list = []

    # 3-2. build X set, which contains 60 grades and gdelt_arr of each grade
    for dim1_idx in range(int(org_X.shape[0]/1)):
        feat_list = []
        # print(str(dim1_idx)+"//"+str(org_X.shape[0]))
        for dim2_idx in range(history_len):
            # 1st column : grade
            col_idx = 0
            X[dim1_idx,dim2_idx*(news_arr_len)+col_idx] = org_X[dim1_idx, dim2_idx+1]
            feat_list.append('ctry_grd_'+str(dim2_idx))
            col_idx = col_idx + 1

            # 2nd, 3rd column : get_avg_gdelt_score in up_features/down_features which read_mode = 1
            X[dim1_idx,dim2_idx*(news_arr_len)+col_idx] = max(0.0,get_normzed_avg_gdelt_score(Sum, m_up_feat_1, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+1]), org_X[dim1_idx, 0], gdelt_idx+"_up", '1', norm_prd_cnt))
            # X[dim1_idx,dim2_idx*(news_arr_len)+col_idx] = max(0.0,get_avg_gdelt_score(Sum, m_up_feat_1, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+1]), org_X[dim1_idx, 0], gdelt_idx+"_up", '1'))
            feat_list.append('up_gdelt_score_'+str(dim2_idx))
            col_idx = col_idx + 1
            X[dim1_idx,dim2_idx*(news_arr_len)+col_idx] = min(0.0,get_normzed_avg_gdelt_score(Sum, m_down_feat_1, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+1]), org_X[dim1_idx, 0], gdelt_idx+"_down", '1', norm_prd_cnt))
            feat_list.append('down_gdelt_score_'+str(dim2_idx))
            col_idx = col_idx + 1

            # next column : get_avg_gdelt_score which read_mode = 2
            if read_mode == '2':
                X[dim1_idx,dim2_idx*(news_arr_len)+col_idx] = max(0.0,get_normzed_avg_gdelt_score(Sum1, m_up_feat_2, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+1]), org_X[dim1_idx, 0], gdelt_idx+"_up", '2', norm_prd_cnt))
                feat_list.append('up_gdelt_obj_score_'+str(dim2_idx))
                col_idx = col_idx + 1
                X[dim1_idx,dim2_idx*(news_arr_len)+col_idx] = min(0.0,get_normzed_avg_gdelt_score(Sum1, m_down_feat_2, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+1]), org_X[dim1_idx, 0], gdelt_idx+"_down", '2', norm_prd_cnt))
                feat_list.append('down_gdelt_obj_score_'+str(dim2_idx))
                col_idx = col_idx + 1

            # next column : get_indicator_score
            if bool_indicators == 1:
                X[dim1_idx,dim2_idx*(news_arr_len)+col_idx] = get_indicator_score(normalized_indic, org_X[dim1_idx, dim2_idx+history_len+1], org_X[dim1_idx, 0])
                feat_list.append('economic_score_'+str(dim2_idx))
                col_idx = col_idx + 1

    # X_df = pd.DataFrame(data=X, columns=feat_list)
    print(X)

    return X, feat_list

k_model = keras.models.load_model("./model/final_"+idx+"_model.h5")
# print(k_model)
pdp_ds, feat_list = build_dataset()
print(max(pdp_ds[:,2]), min(pdp_ds[:,2]))

predict_fn = lambda x: k_model.predict(x.reshape(x.shape[0], history_len, 4))
# .argmax(axis=1)

lr_ale = ALE(predict_fn, feature_names=feat_list, target_names=['-1','0','1'])
# lr_ale = ALE(predict_fn, feature_names=feat_list, target_names=['0','1','2','3','4','5','6','7'])
lr_exp = lr_ale.explain(pdp_ds)
# print(lr_exp.feature_names)
# print(lr_exp.ale_values)
# print(lr_exp.ale0)
# print(lr_exp.feature_deciles)
# , n_cols=2, fig_kw={'figwidth':10, 'figheight': 10}, sharey=None
plot_ale(lr_exp, n_cols=4, fig_kw={'figwidth': 10, 'figheight': 10});
plt.show()

"""
print(tmp_1)
print(tmp_1.ravel())
"""