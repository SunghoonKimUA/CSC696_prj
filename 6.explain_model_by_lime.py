# Explainability with LIME for single prediction
import pandas as pd
from alibi.explainers import ALE, plot_ale

import matplotlib

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

S = pd.read_csv("./data/train_timeseries_10_normed_"+str(history_len)+"_evaldt_df.csv", index_col=0)

def build_dataset():
    S_indic = pd.read_csv("./data/imf_df.csv", index_col=0, low_memory=False)

    # 0. normalizing economic_indicator data
    S_indic = S_indic.replace(99999,0)
    normalized_indic=(S_indic-S_indic.mean())/S_indic.std()
    normalized_indic['ctry_cd'] = S_indic['ctry_cd']
    normalized_indic['dt_dtr']  = S_indic['dt_dtr']

    date_range_lst = pd.read_csv("./data/cre-crc-historical-internet-english.csv", index_col=0).columns[2:]

    # org_X = S[[col for col in S.columns if not col.endswith(str(history_len))]].to_numpy()
    org_X = S.to_numpy()

    Sum, Sum1 = open_summary_csv(var1_idx, var2_idx, var3_idx)
    gdelt_idx = var1_idx+"_"+var2_idx+"_"+var3_idx
    m_feat_1 = get_major_features(var1_idx, var2_idx, var3_idx, 1, num_of_evnt_cd)
    m_feat_2 = get_major_features(var1_idx, var2_idx, var3_idx, 2, num_of_evnt_cd)
    m_up_feat_1, m_down_feat_1 = get_major_features_sep(var1_idx, var2_idx, var3_idx, 1, num_of_evnt_cd)
    m_up_feat_2, m_down_feat_2 = get_major_features_sep(var1_idx, var2_idx, var3_idx, 2, num_of_evnt_cd)

    # 3-1. Estimate new axis's length, grade(1) + new_avg_score(if read_mode == 1 then 1 else 2)
    news_arr_len = 3
    if read_mode == '2':
        news_arr_len = news_arr_len + 2
    if bool_indicators == 1:
        news_arr_len = news_arr_len + 1
    X = np.ndarray(shape=(org_X.shape[0], history_len, news_arr_len))
    feat_list = []
    cate_feat_list = []

    # 3-2. build X set, which contains 60 grades and gdelt_arr of each grade
    for dim1_idx in range(org_X.shape[0]):
        feat_list = []
        cate_feat_list = []
        # print(str(dim1_idx)+"//"+str(org_X.shape[0]))
        for dim2_idx in range(history_len):
            # 1st column : grade
            col_idx = 0
            X[dim1_idx,dim2_idx,col_idx] = org_X[dim1_idx, dim2_idx+1]
            feat_list.append('ctry_grd')
            cate_feat_list.append('ctry_grd')
            col_idx = col_idx + 1
            # 2nd column : get_avg_gdelt_score which read_mode = 1
            X[dim1_idx,dim2_idx,col_idx] = max(0.0,get_avg_gdelt_score(Sum, m_up_feat_1, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+3]), org_X[dim1_idx, 0], gdelt_idx+"_up", '1'))
            feat_list.append('up_gdelt_score')
            col_idx = col_idx + 1
            X[dim1_idx,dim2_idx,col_idx] = min(0.0,get_avg_gdelt_score(Sum, m_down_feat_1, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+3]), org_X[dim1_idx, 0], gdelt_idx+"_down", '1'))
            feat_list.append('down_gdelt_score')
            col_idx = col_idx + 1
            # next column : get_avg_gdelt_score which read_mode = 2
            if read_mode == '2':
                X[dim1_idx,dim2_idx,col_idx] = max(0.0,get_avg_gdelt_score(Sum1, m_up_feat_2, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+3]), org_X[dim1_idx, 0], gdelt_idx+"_up", '2'))
                feat_list.append('up_gdelt_obj_score')
                col_idx = col_idx + 1
                X[dim1_idx,dim2_idx,col_idx] = min(0.0,get_avg_gdelt_score(Sum1, m_down_feat_2, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+3]), org_X[dim1_idx, 0], gdelt_idx+"_down", '2'))
                feat_list.append('down_gdelt_obj_score')
                col_idx = col_idx + 1

            # next column : get_indicator_score
            if bool_indicators == 1:
                X[dim1_idx,dim2_idx,col_idx] = get_indicator_score(normalized_indic, org_X[dim1_idx, dim2_idx+history_len+3], org_X[dim1_idx, 0])
                feat_list.append('economic_score')
                col_idx = col_idx + 1

    # X_df = pd.DataFrame(data=X, columns=feat_list)
    # print(X_df)

    return X, feat_list, cate_feat_list

def get_ds_idx(S, ctry_name, eval_dt):
    tmp_row = S[(S["ISO"]==ctry_name) & (S["evaldt_"+str(history_len).zfill(2)]==eval_dt)]
    return tmp_row.index[0], (tmp_row.iat[0,history_len], tmp_row.iat[0,history_len+1])

k_model = keras.models.load_model("./model/final_"+idx+"_model.h5")
# print(k_model)
pdp_ds, feat_list, cate_feat_list = build_dataset()

import lime
import lime.lime_tabular
# ['0','1','2','3','4','5','6','7']
explainer = lime.lime_tabular.RecurrentTabularExplainer(pdp_ds, class_names=['same','up','down'], feature_names=feat_list,
                                                   categorical_features=cate_feat_list, 
                                                   categorical_names="cate_name")
# print(get_ds_idx(S, "RUS", "11-Mar-22"))
"""
predict_fn = lambda x: k_model.predict(x.reshape(x.shape[0], 6, 3))
proba_arr = predict_fn(pdp_ds[i].reshape(1, 18))
print(np.sum(proba_arr))
"""
ds_idx = get_ds_idx(S, "RUS", "11-Mar-22")
print(ds_idx)
#exp = explainer.explain_instance(pdp_ds[ds_idx[0]], k_model.predict, labels=ds_idx[1], num_features=15)
exp = explainer.explain_instance(pdp_ds[ds_idx[0]], k_model.predict, labels=(0,1,2), num_features=10)
exp.save_to_file('./result/lime_0506.html')
"""
ds_idx = get_ds_idx(S, "BLR", "11-Mar-22")
exp = explainer.explain_instance(pdp_ds[ds_idx[0]], k_model.predict, labels=ds_idx[1], num_features=15)
exp.save_to_file('./lime_0423_BLR.html')
"""