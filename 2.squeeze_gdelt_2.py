import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from datetime import datetime

from os import listdir
from os.path import isfile, join

# 4/27 for NumMentions
postfix_d = ""
gdelt_col_idx = 34
"""
postfix_d = "_cnt"
gdelt_col_idx = 31
"""

def build_evnt_csv(var1_idx, var2_idx, var3_idx, data_path, date_range_lst):
    range_idx = 0
    # create range_pivot dataframe with strt_dd, end_dd, ctry_1, ctry_2, each evnt_cd_list''s sum(rt), each evnt_cd_list''s sum(cnt)
    # dd >= strt_dd and dd < end_dd
    last_year = pd.DataFrame()
    for f in listdir(data_path):
        if isfile(join(data_path, f)):
            # print(f, range_idx)

            chunk = pd.read_csv(join(data_path, f), dtype={'dd':str}, index_col=0, low_memory=False)
            # data cleansing
            chunk = chunk.drop(chunk[chunk['dd'] < (f[:4]+'0101')].index)
            chunk = chunk.drop(chunk[chunk['evnt_cd']=='---'].index)

            chunk = pd.concat([last_year, chunk])
            for r_idx in range(range_idx,len(date_range_lst)):
                if range_idx == 0:
                    strt__dt = (datetime.strptime(date_range_lst[range_idx],"%d-%b-%y")-single_day*(180+var1_idx)).strftime("%Y%m%d")
                    end__dt  = (datetime.strptime(date_range_lst[range_idx],"%d-%b-%y")-single_day*var1_idx)      .strftime("%Y%m%d")
                    ## just waste data which is before strt_dd
                    chunk = chunk.drop(chunk[chunk['dd'] < strt__dt].index)
                else:
                    strt__dt = (datetime.strptime(date_range_lst[range_idx-1],"%d-%b-%y")-single_day*var1_idx).strftime("%Y%m%d")
                    end__dt  = (datetime.strptime(date_range_lst[range_idx  ],"%d-%b-%y")-single_day*var1_idx).strftime("%Y%m%d")
                # print(f, strt__dt, end__dt, range_idx, len(chunk), chunk.iat[0,1], chunk.iat[-1,1])

                if f.startswith(end__dt[:4]):
                    trgt_pivot = chunk[chunk['dd'] < end__dt]
                    # print(len(trgt_pivot))
                    p_table = pd.pivot_table(trgt_pivot, values=['MySum', 'MyCount'], index=['ctry_1', 'ctry_2'], columns=['evnt_cd']
                                           , aggfunc={'MySum': np.sum, 'MyCount': np.sum}, fill_value=0.0)
                    p_table.columns = [s1 + str(int(float(s2))) for (s1,s2) in p_table.columns.tolist()]
                    p_table.reset_index(inplace=True)
#                                print(p_table.columns)

                    p_table.to_csv('./data'+postfix_d+'/pivot'+var3_pfix+var2_pfix+'_'+str(var1_idx)+'/'+str(range_idx).zfill(3)+'.csv')
#                    chunk = chunk.drop(chunk[chunk['dd'] < end__dt].index)
#                    print(chunk[chunk['dd'] < end__dt])
                    chunk = chunk[chunk['dd'] >= end__dt]
                    print(end__dt, chunk.iat[0,1], chunk.iat[-1,1])
#                    print(range_idx, trgt_pivot)
                    last_year = pd.DataFrame()
                    range_idx = range_idx+1
                elif f.startswith(strt__dt[:4]):
                    last_year = chunk
                else:
                    continue

def read_csv(data_path, f, last_year, strt__dt):
    chunk = pd.read_csv(join(data_path, f), dtype={'dd':str}, index_col=0, low_memory=False)
    # data cleansing
    chunk = chunk.drop(chunk[chunk['dd'] < (f[:4]+'0101')].index)
    chunk = chunk.drop(chunk[chunk['evnt_cd']=='---'].index)
    chunk = chunk.drop(chunk[chunk['evnt_cd']=='00X'].index)
    chunk = chunk.drop(chunk[chunk['evnt_cd']=='X'].index)

    chunk = pd.concat([last_year, chunk])
    ## just waste data which is before strt_dd
    chunk = chunk.drop(chunk[chunk['dd'] < strt__dt].index)

    return chunk

# build gdelt_csv from end_Dt - 180 days to end_Dt
def build_time_csv(var1_idx, var2_idx, var3_idx, data_path, date_range_lst):
    range_idx = 0
    # exchange the order of loop compared to build_evnt_csv
    # create range_pivot dataframe with strt_dd, end_dd, ctry_1, ctry_2, each evnt_cd_list''s sum(rt), each evnt_cd_list''s sum(cnt)
    # dd >= strt_dd and dd < end_dd
    last_year = pd.DataFrame()
    for range_idx in range(0,len(date_range_lst)):
        strt__dt = (datetime.strptime(date_range_lst[range_idx],"%d-%b-%y")-single_day*(180+var1_idx)).strftime("%Y%m%d")
        end__dt  = (datetime.strptime(date_range_lst[range_idx],"%d-%b-%y")-single_day*var1_idx).strftime("%Y%m%d")

        for f in listdir(data_path):
            if isfile(join(data_path, f)):
                # print(f, range_idx)

                if f.startswith(end__dt[:4]):
                    chunk = read_csv(data_path, f, last_year, strt__dt)

                    trgt_pivot = chunk[chunk['dd'] < end__dt]
                    ## make columns for range_of_date, substr(evnt_cd,1,2) in trgt_pivot
                    #try:
                    trgt_pivot['evnt_2cd'] = trgt_pivot['evnt_cd'].apply(lambda x: str(int(float(x))).zfill(3)[:2])
                    #except:
                    #    print(f)
                    #    exit
                    trgt_pivot['range_of_dt'] = trgt_pivot['dd'].apply(lambda x: int((datetime.strptime(end__dt,'%Y%m%d') - datetime.strptime(x,'%Y%m%d') - single_day)/(30*single_day)))
                    p_table = pd.pivot_table(trgt_pivot, values=['MySum', 'MyCount'], index=['ctry_1', 'ctry_2'], columns=['range_of_dt','evnt_2cd']
                                           , aggfunc={'MySum': np.sum, 'MyCount': np.sum}, fill_value=0.0)
                    p_table.columns = [s1 + '_' + str(s2) + '_' + s3 for (s1,s2,s3) in p_table.columns.tolist()]

                    p_table.reset_index(inplace=True)
    #                                print(p_table.columns)

                    p_table.to_csv('./data'+postfix_d+'/pivot'+var3_pfix+var2_pfix+'_'+str(var1_idx)+'/'+str(range_idx).zfill(3)+'.csv')
                    last_year = pd.DataFrame()
                elif f.startswith(strt__dt[:4]):
                    chunk = read_csv(data_path, f, last_year, strt__dt)
                    last_year = chunk
                else:
                    continue


agg_data = []
data_path = 'e://tmp/'
base_yy = '1998'

# 1. select sum(avg_tone), count(*) # avg_tone=34
#           sum(NumMentions)  ## 4/27 (NumMentions=31)
#      from gdelt_1_0
#     where dd >= 19980101
#     group by dd, ctry_1, ctry_2, evnt_cd
# 2 options.
#       dir=gdelt_1_0_nm means summation of all events,
#   and dir=gdelt_1_0_ex means summation of events.avgtone <= -10 or events.avgtone >= 10
"""
gen_options = [0,1]
for opt in gen_options:
    base_yy = '1998'
    for f in listdir(data_path):
        if isfile(join(data_path, f)):
            for chunk in  pd.read_csv(join(data_path, f), sep='\t',low_memory=False, index_col=0, header=None, chunksize=200000):
                chunk = chunk.drop(chunk[chunk[3] != int(f[:4])].index)
                chunk = chunk.drop(chunk[chunk[27].isna()==True].index)
                if opt == 1:
                    chunk = chunk.drop(chunk[chunk[gdelt_col_idx].between(-10,10)].index)
                agg_data.append(chunk
                                .groupby(by=[1, 7, 17, 27],dropna=False)[gdelt_col_idx].agg(MySum='sum', MyCount='count')
                                .reset_index()
                                .rename(columns={1: "dd", 7: "ctry_1", 17: "ctry_2", 27: "evnt_cd"}))

            if base_yy != f[:4]:
                agg_df = pd.concat(agg_data, axis=0)
                print(agg_df.columns)

                agg_this_df = agg_df[agg_df.dd.between(int(base_yy+'0101'), int(base_yy+'1231'))]
                agg_next_df = agg_df[agg_df.dd > int(base_yy+'1231')]

                agg_this_df = agg_this_df.reset_index()
                agg_this_df.to_csv('./data'+postfix_d+'/gdelt_1_0'+('_ex' if opt == 1 else '_nm')+'/'+base_yy+'.csv')
                base_yy = str(int(base_yy)+1)
                agg_data = []
                agg_data.append(agg_next_df)

    # for the last year
    agg_df = pd.concat(agg_data, axis=0)
    agg_df = agg_df.reset_index()
    agg_df.to_csv('./data'+postfix_d+'/gdelt_1_0'+('_ex' if opt == 1 else '_nm')+'/'+base_yy+'.csv')
"""
"""
# 1-old. test version of 1
base_yy = '201801'

for f in listdir(data_path):
    if isfile(join(data_path, f)):
        for chunk in  pd.read_csv(join(data_path, f), sep='\t',low_memory=False, index_col=0, header=None, chunksize=200000):
            ## data cleansing
            chunk = chunk.drop(chunk[chunk[3] != int(f[:4])].index)
            chunk = chunk.drop(chunk[chunk[27].isna()==True].index)
            agg_data.append(chunk.groupby(by=[1, 7, 17, 27],dropna=False)[gdelt_col_idx].agg(MySum='sum', MyCount='count'))
        #    print(chunk.groupby(by=[1, 7, 17, 27])[gdelt_col_idx].agg(MySum='sum', MyCount='count'))
#        print(f)
        if base_yy != f[:6]:
            agg_df = pd.concat(agg_data, axis=0)
            agg_df.to_csv('./data/gdelt_1_0/'+base_yy+'.csv')
            if base_yy.endswith('12'):
                base_yy = str(int(base_yy[:4])+1)+'01'
            else:
                base_yy = str(int(base_yy)+1)

"""

S = pd.read_csv("./data/cre-crc-historical-internet-english.csv", index_col=0)
date_range_lst = S.columns[2:]
print(date_range_lst)

from datetime import timedelta
single_day = timedelta(days=1)
# shift eval_period (ex. 22-Jan-99 -> 22-Jan-99, 17-Jan-99, 12-Jan-99, 7-Jan-99, 2-Jan-99 
# var1 = [0, 5, 10, 15, 20]
var1 = [0, 5, 10, 15, 20, 30]# 
# normal : all events, extra : events.avgtone <= -10 or events.avgtone >= 10
var2 = ['normal','extra']#
var2_pfix = ''
# 1. evnt_cd : strt_dt ~ end_dt. group by evnt_cd
# 2. time_series : end_dt-180 ~ end_dt. group by substr(evnt_cd,1,2)
# var3 = ['evnt_cd','time_series']
var3 = ['evnt_cd']
var3_pfix = ''

for var1_idx in var1:
    for var2_idx in var2:
        if var2_idx == 'extra':
            var2_pfix = '_ex'
        else:
            var2_pfix = '_nm'
        data_path = './data'+postfix_d+'/gdelt_1_0'+var2_pfix+'/'

        for var3_idx in var3:
            if var3_idx == 'evnt_cd':
                var3_pfix = '_ev'
                build_evnt_csv(var1_idx, var2_idx, var3_idx, data_path, date_range_lst)
            else:
                var3_pfix = '_ts'
                build_time_csv(var1_idx, var2_idx, var3_idx, data_path, date_range_lst)
"""
# 3. build time_series data to every first evaluation date, since first evaluation date has no previous eval_date
var3 = ['evnt_cd','time_series']
for var1_idx in var1:
    for var2_idx in var2:
        if var2_idx == 'extra':
            var2_pfix = '_ex'
        else:
            var2_pfix = '_nm'
        data_path = './data/gdelt_1_0'+var2_pfix+'/'

        for var3_idx in var3:
            if var3_idx == 'evnt_cd':
                var3_pfix = '_ev'
                build_evnt_csv(var1_idx, var2_idx, var3_idx, data_path, date_range_lst[:1])
            else:
                var3_pfix = '_ts'
                # build_time_csv(var1_idx, var2_idx, var3_idx, data_path, date_range_lst[:1])
"""

"""
# check that all pivot table have same columns
data_path = './data/pivot/'
for f in listdir(data_path):
    if isfile(join(data_path, f)):
        print(f)
        chunk = pd.read_csv(join(data_path, f))
        print(chunk.columns)
# no!
"""
