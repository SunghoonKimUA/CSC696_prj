import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# 1. read "cre-crc-historical-internet-english.csv"
S = pd.read_csv("./data/cre-crc-historical-internet-english.csv", index_col=0)
end_dt_lst = S.columns[2:]
# print(end_dt_lst)

# 2. choose target countries
S_real = S[S.iloc[:,89].str.match(r'[0-7]')==True].copy()
# print(S_real["ISO"])

# 3. build timeseries datasets
timeseries_col = 60+1
timeseries_ds = pd.DataFrame()
col_arr = []
"""
for idx in range(timeseries_col):
    col_arr.append("dt_"+str(idx).zfill(2))

for col_idx in range(2, len(S_real.columns)):
    # print(col_idx)
    if col_idx < len(S_real.columns) - timeseries_col + 1:
        tmp_ds = S_real.iloc[:,col_idx:col_idx+timeseries_col]
        tmp_ds.columns = col_arr
        tmp_ds = pd.concat([S_real["ISO"], tmp_ds], axis=1)
        # if col_idx == 2:
        #     print(S_real["ISO"].shape, tmp_ds.shape)
        # print(tmp_ds)
        timeseries_ds = pd.concat([timeseries_ds, tmp_ds])

# before drop '-' grade : 3973 rows
timeseries_ds = timeseries_ds.replace('-',None)
# print(len(timeseries_ds))
timeseries_ds = timeseries_ds.dropna()
# print(len(timeseries_ds))
# after drop '-' grade : 3458 rows
timeseries_ds = timeseries_ds.reset_index()
# print(timeseries_ds.columns)
timeseries_ds = timeseries_ds.drop(['Country Name (1)'], axis=1)
timeseries_ds.to_csv('./data/train_timeseries_df.csv')

# 3-1. build timeseries + eval_date datasets
timeseries1_ds = pd.DataFrame()
col_evaldt_arr = []

for idx in range(timeseries_col):
    col_evaldt_arr.append("evaldt_"+str(idx).zfill(2))

for col_idx in range(2, len(S_real.columns)):
    # print(col_idx)
    if col_idx < len(S_real.columns) - timeseries_col + 1:
        tmp_ds = S_real.iloc[:,col_idx:col_idx+timeseries_col]
        tmp_ds.columns = col_arr
        tmp_header = S_real.columns[col_idx:col_idx+timeseries_col]
        tmp_header = pd.DataFrame([tmp_header]*len(tmp_ds))
        tmp_header.columns = col_evaldt_arr
        tmp_ds = pd.concat([S_real["ISO"].reset_index(), tmp_ds.reset_index(), tmp_header.reset_index()], axis=1)
        timeseries1_ds = pd.concat([timeseries1_ds, tmp_ds])

# before drop '-' grade : 3973 rows
timeseries1_ds = timeseries1_ds.replace('-',None)
# print(len(timeseries1_ds))
timeseries1_ds = timeseries1_ds.dropna()
# print(len(timeseries1_ds))
# after drop '-' grade : 3459 rows
timeseries1_ds = timeseries1_ds.reset_index()
timeseries1_ds = timeseries1_ds.drop(['Country Name (1)'], axis=1)
timeseries1_ds = timeseries1_ds.drop(['level_0'], axis=1)
timeseries1_ds = timeseries1_ds.drop(['index'], axis=1)
timeseries1_ds.to_csv('./data/train_timeseries_evaldt_df.csv')

# 3-2. build timeseries + eval_date datasets to choose optimal history_length
timeseries_cand = [60, 36, 12, 6, 3]
for tmp_len in timeseries_cand:
    timeseries2_ds = pd.DataFrame()
    col_arr = []
    col_evaldt_arr = []

    # to concatenate "y" column, plus 1
    for idx in range(tmp_len+1):
        col_arr.append("dt_"+str(idx).zfill(2))

    for idx in range(tmp_len+1):
        col_evaldt_arr.append("evaldt_"+str(idx).zfill(2))

    for col_idx in range(2, len(S_real.columns)):
        # print(col_idx)
        if col_idx < len(S_real.columns) - tmp_len:
            tmp_ds = S_real.iloc[:,col_idx:col_idx+tmp_len+1]
            tmp_ds.columns = col_arr
            tmp_header = S_real.columns[col_idx:col_idx+tmp_len+1]
            tmp_header = pd.DataFrame([tmp_header]*len(tmp_ds))
            tmp_header.columns = col_evaldt_arr

            tmp_ds = pd.concat([S_real["ISO"].reset_index(), tmp_ds.reset_index(), tmp_header.reset_index()], axis=1)
            timeseries2_ds = pd.concat([timeseries2_ds, tmp_ds])

    timeseries2_ds = timeseries2_ds.replace('-',None)
    timeseries2_ds = timeseries2_ds.dropna()
    timeseries2_ds = timeseries2_ds.reset_index()
    timeseries2_ds = timeseries2_ds.drop(['Country Name (1)'], axis=1)
    timeseries2_ds = timeseries2_ds.drop(['level_0'], axis=1)
    timeseries2_ds = timeseries2_ds.drop(['index'], axis=1)
    timeseries2_ds.to_csv('./data/train_timeseries_'+str(tmp_len)+'_evaldt_df.csv')
"""

# 3-3. build timeseries + eval_date datasets to choose optimal history_length and to normalize blank first 10 or 5 columns
timeseries_cand = [60, 36, 12, 6, 3]
normze_prd_cand = [5, 10]
for tmp_nor_len in normze_prd_cand:
    for tmp_len in timeseries_cand:
        timeseries3_ds = pd.DataFrame()
        col_arr = []
        col_evaldt_arr = []

        # to concatenate "y" column, plus 1
        for idx in range(tmp_len+1):
            col_arr.append("dt_"+str(idx).zfill(2))

        for idx in range(tmp_len+1):
            col_evaldt_arr.append("evaldt_"+str(idx).zfill(2))

        for col_idx in range(2+tmp_nor_len-1, len(S_real.columns)):
            # print(col_idx)
            if col_idx < len(S_real.columns) - tmp_len:
                tmp_ds = S_real.iloc[:,col_idx:col_idx+tmp_len+1]
                tmp_ds.columns = col_arr
                tmp_header = S_real.columns[col_idx:col_idx+tmp_len+1]
                tmp_header = pd.DataFrame([tmp_header]*len(tmp_ds))
                tmp_header.columns = col_evaldt_arr

                tmp_ds = pd.concat([S_real["ISO"].reset_index(), tmp_ds.reset_index(), tmp_header.reset_index()], axis=1)
                timeseries3_ds = pd.concat([timeseries3_ds, tmp_ds])

        timeseries3_ds = timeseries3_ds.replace('-',None)
        timeseries3_ds = timeseries3_ds.dropna()
        timeseries3_ds = timeseries3_ds.reset_index()
        timeseries3_ds = timeseries3_ds.drop(['Country Name (1)'], axis=1)
        timeseries3_ds = timeseries3_ds.drop(['level_0'], axis=1)
        timeseries3_ds = timeseries3_ds.drop(['index'], axis=1)
        timeseries3_ds.to_csv('./data/train_timeseries_'+str(tmp_nor_len)+'_normed_'+str(tmp_len)+'_evaldt_df.csv')
