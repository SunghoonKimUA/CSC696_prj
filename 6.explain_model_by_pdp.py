import pandas as pd
from tensorflow import keras

import numpy as np
from ds_lib import *

# ev_nm_15_1_10_1_gru__6
var1_idx = 'ev'
var2_idx = 'nm'
var3_idx = '15'
read_mode = '1'
num_of_evnt_cd = 10
bool_indicators = 1
idx = build_idx(var1_idx, var2_idx, var3_idx, read_mode, num_of_evnt_cd, bool_indicators)+"_gru__6"

def build_dataset():
    S_indic = pd.read_csv("./data/imf_df.csv", index_col=0, low_memory=False)

    # 0. normalizing economic_indicator data
    S_indic = S_indic.replace(99999,0)
    normalized_indic=(S_indic-S_indic.mean())/S_indic.std()
    normalized_indic['ctry_cd'] = S_indic['ctry_cd']
    normalized_indic['dt_dtr']  = S_indic['dt_dtr']

    history_len = 6
    S = pd.read_csv("./data/train_timeseries_"+str(history_len)+"_evaldt_df.csv", index_col=0)
    date_range_lst = pd.read_csv("./data/cre-crc-historical-internet-english.csv", index_col=0).columns[2:]

    org_X = S[[col for col in S.columns if not col.endswith(str(history_len))]].to_numpy()

    Sum, Sum1 = open_summary_csv(var1_idx, var2_idx, var3_idx)
    gdelt_idx = var1_idx+"_"+var2_idx+"_"+var3_idx
    m_feat_1 = get_major_features(var1_idx, var2_idx, var3_idx, 1, num_of_evnt_cd)
    m_feat_2 = get_major_features(var1_idx, var2_idx, var3_idx, 2, num_of_evnt_cd)

    # 3-1. Estimate new axis's length, grade(1) + new_avg_score(if read_mode == 1 then 1 else 2)
    news_arr_len = 2
    if read_mode == '2':
        news_arr_len = news_arr_len + 1
    if bool_indicators == 1:
        news_arr_len = news_arr_len + 1
    X = np.ndarray(shape=(org_X.shape[0], history_len*news_arr_len))
    feat_list = []

    # 3-2. build X set, which contains 60 grades and gdelt_arr of each grade
    for dim1_idx in org_X.shape[0]:
        feat_list = []
        # print(str(dim1_idx)+"//"+str(org_X.shape[0]))
        for dim2_idx in range(history_len):
            # 1st column : grade
            col_idx = 0
            X[dim1_idx,dim2_idx*(news_arr_len)+col_idx] = org_X[dim1_idx, dim2_idx+1]
            feat_list.append('ctry_grd_'+str(dim2_idx))
            col_idx = col_idx + 1
            # 2nd column : get_avg_gdelt_score which read_mode = 1
            X[dim1_idx,dim2_idx*(news_arr_len)+col_idx] = get_avg_gdelt_score(Sum, m_feat_1, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+1]), org_X[dim1_idx, 0], gdelt_idx, '1')
            feat_list.append('gdelt_score_'+str(dim2_idx))
            col_idx = col_idx + 1

            # next column : get_avg_gdelt_score which read_mode = 2
            if read_mode == '2':
                X[dim1_idx,dim2_idx*(news_arr_len)+col_idx] = get_avg_gdelt_score(Sum1, m_feat_2, date_range_lst.get_loc(org_X[dim1_idx, dim2_idx+history_len+1]), org_X[dim1_idx, 0], gdelt_idx, '2')
                feat_list.append('gdelt_score_obj_'+str(dim2_idx))
                col_idx = col_idx + 1

            # next column : get_indicator_score
            if bool_indicators == 1:
                X[dim1_idx,dim2_idx*(news_arr_len)+col_idx] = get_indicator_score(normalized_indic, org_X[dim1_idx, dim2_idx+history_len+1], org_X[dim1_idx, 0])
                feat_list.append('economic_score_'+str(dim2_idx))
                col_idx = col_idx + 1

    print(X[:2])
    X_df = pd.DataFrame(data=X, columns=feat_list)
    # print(X_df)

    return X_df, feat_list

k_model = keras.models.load_model("./model/best_timeseries_"+idx+"_model.h5")
print("./model/best_timeseries_"+idx+"_model.h5")
# print(k_model)
pdp_df, feat_list = build_dataset()
print(pdp_df.shape)
"""
from pdpbox import pdp

# features = ["gdelt_score_5", "economic_score_5", ("gdelt_score_5", "economic_score_5")]
pdp_combo_rf = pdp.pdp_interact(
    model=k_model, dataset=pdp_df, model_features=feat_list, features=["gdelt_score_5", "economic_score_5"], 
    num_grid_points=[10, 10],  percentile_ranges=[None, None]
)
# stop because of message, "ValueError: Shape of passed values is (10569, 8), indices imply (10569, 1)"
fig, axes = pdp.pdp_interact_plot(
    pdp_combo_rf, ["gdelt_score_5", "economic_score_5"], plot_type='contour', x_quantile=True, ncols=2, 
    plot_pdp=True, which_classes=[0, 1, 2, 3]
)

"""
import matplotlib.pyplot as plt

x = pd.concat([pdp_df.iloc[:,1],pdp_df.iloc[:,4],pdp_df.iloc[:,7],pdp_df.iloc[:,10],pdp_df.iloc[:,13],pdp_df.iloc[:,16]]).to_numpy()
y = pd.concat([pdp_df.iloc[:,2],pdp_df.iloc[:,5],pdp_df.iloc[:,8],pdp_df.iloc[:,11],pdp_df.iloc[:,14],pdp_df.iloc[:,17]]).to_numpy()

area = 2

plt.scatter(x, y, s=area, c='black', alpha=0.5)
plt.show()

# there is certain correlation between news vs economic_indicator
# pdpbox : not allow to one-hot encoding result
# scikit-learn : ask to "fit" however it was "fitted" by keras.
