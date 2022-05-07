import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime

from os import listdir
from os.path import isfile, join

data_path = './data/pivot_'
def open_summary_csv(var1_idx, var2_idx, var3_idx):
    S = pd.read_csv(data_path+var1_idx+"_"+var2_idx+"_"+var3_idx+"/summary_1.csv", index_col=0, low_memory=False)
    S1 = pd.read_csv(data_path+var1_idx+"_"+var2_idx+"_"+var3_idx+"/summary_2.csv", index_col=0, low_memory=False)
    return S, S1

def save_major_features(up_down_ds, gdelt_ds, path, file_nm):
    tmp_ds = []
    for r2_idx in range(len(up_down_ds)):
        # print(up_ds.iloc[r2_idx,0], end_dt_lst.get_loc(up_ds.iloc[r2_idx,3]))
        # print(S[(S['ctry_cd'] == up_ds.iloc[r2_idx,0]) & (S['file_idx'] == end_dt_lst.get_loc(up_ds.iloc[r2_idx,3]))].iloc[0,2:])
        row_arr = gdelt_ds[(gdelt_ds['ctry_cd'] == up_down_ds.iloc[r2_idx,0]) & (gdelt_ds['file_idx'] == end_dt_lst.get_loc(up_down_ds.iloc[r2_idx,3]))]
        if len(row_arr)>0:
            tmp_ds.append(row_arr.iloc[0,2:])

    tmp_df = pd.DataFrame(tmp_ds)
    # print(tmp_df)

    evnt_cd_ds = []
    for evnt_cd in range(10,204):
        evnt_cd_ds.append([evnt_cd, tmp_df['MyCount'+str(evnt_cd)].sum(), tmp_df['MySum'+str(evnt_cd)].sum(), tmp_df['MySum'+str(evnt_cd)].sum()/tmp_df['MyCount'+str(evnt_cd)].sum()])
    evnt_cd_df = pd.DataFrame(evnt_cd_ds, columns=['evnt_cd', 'sum_count', 'sum_score', 'avg_score']).fillna(0.0)
    print(evnt_cd_df)
    evnt_cd_df.to_csv(join(data_path+path,file_nm))


# 1. read "cre-crc-historical-internet-english.csv"
S = pd.read_csv("./data/cre-crc-historical-internet-english.csv", index_col=0)
end_dt_lst = S.columns[2:]
# print(end_dt_lst)

# 2. choose up/down grades
up_arr = []
down_arr = []
for r_idx in range(len(S)):
    for idx in range(3, len(S.columns)):
        if S.iloc[r_idx, idx-1] != '-' and S.iloc[r_idx, idx] != '-' and S.iloc[r_idx, idx-1] > S.iloc[r_idx, idx]:
            # print('down', S.iloc[r_idx, 0], S.iloc[r_idx, idx-1], S.iloc[r_idx, idx], S.columns[idx])
            down_arr.append([S.iloc[r_idx, 0], S.iloc[r_idx, idx-1], S.iloc[r_idx, idx], S.columns[idx]])
        if S.iloc[r_idx, idx-1] != '-' and S.iloc[r_idx, idx] != '-' and S.iloc[r_idx, idx-1] < S.iloc[r_idx, idx]:
            # print('up', S.iloc[r_idx, 0], S.iloc[r_idx, idx-1], S.iloc[r_idx, idx], S.columns[idx])
            up_arr.append([S.iloc[r_idx, 0], S.iloc[r_idx, idx-1], S.iloc[r_idx, idx], S.columns[idx]])

up_ds = pd.DataFrame(up_arr, columns=['ctry_cd', 'prev_grd', 'curr_grd', 'eval_dt'])
down_ds = pd.DataFrame(down_arr, columns=['ctry_cd', 'prev_grd', 'curr_grd', 'eval_dt'])
# print(down_ds, up_ds)


# 3. read gdelt summary dataset at each time_slot
import itertools

option_lists = [
#   ['ev', 'ts'],
   ['ev'],
   ['ex', 'nm'],
   ['0', '5', '10', '15', '20', '30'],
#   ['1', '2']
]
experiment_arr = itertools.product(*option_lists)

for var1_idx, var2_idx, var3_idx in experiment_arr:
    S, S1 = open_summary_csv(var1_idx, var2_idx, var3_idx)
    path = var1_idx+"_"+var2_idx+"_"+var3_idx

    # up_ds and action1Code
    save_major_features(up_ds, S, path, 'up_features_1.csv')
    # up_ds and action2Code
    save_major_features(up_ds, S1, path, 'up_features_2.csv')
    # down_ds and action1Code
    save_major_features(down_ds, S, path, 'down_features_1.csv')
    # down_ds and action2Code
    save_major_features(down_ds, S1, path, 'down_features_2.csv')
