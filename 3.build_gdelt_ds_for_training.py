import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from datetime import datetime
from os import listdir
from os.path import isfile, join
import itertools

# aggregate directory's csv into one csv
# key = file_idx, ctry_cd
# value = sum by pivot

data_path = './data/pivot_'
option_lists = [
   ['ev'], # , 'ts'
   ['nm'], # 'ex', 
#   ['ts'],
#   ['nm'],
   ['10', '15', '20', '30'], # '0', '5', 
   ['1', '2']
]
experiment_arr = itertools.product(*option_lists)

for var1_idx, var2_idx, var3_idx, read_mode in experiment_arr:
    experi_path = data_path+var1_idx+"_"+var2_idx+"_"+var3_idx+"/"
    aggr_df1 = pd.DataFrame()
    aggr_df2 = pd.DataFrame()
    print(experi_path)

    for f in listdir(experi_path):
        if isfile(join(experi_path, f)) and f.endswith('csv') and f.startswith('0'):
            # print(f[:3])
            f_header = f.split('_',2)[0]
            trgt_pivot = pd.read_csv(join(experi_path, f), low_memory=False, index_col=0)
            col_name = []
            if var1_idx == 'ev':
                col_name = np.concatenate((['ctry_cd'],['MyCount'+str(evnt_cd) for evnt_cd in range(10,204)],['MySum'+str(evnt_cd) for evnt_cd in range(10,204)]))
            else:
                col_name = np.concatenate((['ctry_cd']
                            ,['MyCount_0_'+str(evnt_cd) for evnt_cd in range(10,204)],['MySum_0_'+str(evnt_cd) for evnt_cd in range(10,204)]
                            ,['MyCount_1_'+str(evnt_cd) for evnt_cd in range(10,204)],['MySum_1_'+str(evnt_cd) for evnt_cd in range(10,204)]
                            ,['MyCount_2_'+str(evnt_cd) for evnt_cd in range(10,204)],['MySum_2_'+str(evnt_cd) for evnt_cd in range(10,204)]
                            ,['MyCount_3_'+str(evnt_cd) for evnt_cd in range(10,204)],['MySum_3_'+str(evnt_cd) for evnt_cd in range(10,204)]
                            ,['MyCount_4_'+str(evnt_cd) for evnt_cd in range(10,204)],['MySum_4_'+str(evnt_cd) for evnt_cd in range(10,204)]
                            ,['MyCount_5_'+str(evnt_cd) for evnt_cd in range(10,204)],['MySum_5_'+str(evnt_cd) for evnt_cd in range(10,204)]
                                          ))

            if read_mode == '1':
                ctry_col_name = 'ctry_1'
            else:
                ctry_col_name = 'ctry_2'

            ctry_arr = np.unique(trgt_pivot[ctry_col_name])
            pivot_arr = np.ndarray(shape=(len(ctry_arr),len(col_name)-1))
            row_idx = 0
            for ctry_cd in ctry_arr:
                tmp_row = []
                if var1_idx == 'ev':
                    for evnt_cd in range(10,204):
                        if 'MyCount'+str(evnt_cd) in trgt_pivot.columns:
                            tmp_row.append(trgt_pivot.loc[(trgt_pivot[ctry_col_name] == ctry_cd), 'MyCount'+str(evnt_cd)].sum())
                        else:
                            tmp_row.append(0.0)
                    for evnt_cd in range(10,204):
                        if 'MySum'+str(evnt_cd) in trgt_pivot.columns:
                            tmp_row.append(trgt_pivot.loc[(trgt_pivot[ctry_col_name] == ctry_cd), 'MySum'+str(evnt_cd)].sum())
                        else:
                            tmp_row.append(0.0)
                else:
                    for ts_idx in range(0,6):
                        for evnt_cd in range(10,204):
                            if 'MyCount_'+str(ts_idx)+'_'+str(evnt_cd) in trgt_pivot.columns:
                                tmp_row.append(trgt_pivot.loc[(trgt_pivot[ctry_col_name] == ctry_cd), 'MyCount_'+str(ts_idx)+'_'+str(evnt_cd)].sum())
                            else:
                                tmp_row.append(0.0)
                        for evnt_cd in range(10,204):
                            if 'MySum_'+str(ts_idx)+'_'+str(evnt_cd) in trgt_pivot.columns:
                                tmp_row.append(trgt_pivot.loc[(trgt_pivot[ctry_col_name] == ctry_cd), 'MySum_'+str(ts_idx)+'_'+str(evnt_cd)].sum())
                            else:
                                tmp_row.append(0.0)
                pivot_arr[row_idx] = tmp_row
                row_idx = row_idx+1

            p_table = pd.DataFrame(data=np.concatenate([ctry_arr.reshape(ctry_arr.shape[0],1),pivot_arr],axis=1), columns=col_name)
            """ failed to apply pivot
            p_table.reset_index(inplace=True)
            if read_mode == '1':
                p_table = pd.pivot_table(trgt_pivot, index=['ctry_1'], aggfunc=sum, fill_value=0.0)
            else:
                p_table = pd.pivot_table(trgt_pivot, index=['ctry_2'], aggfunc=sum, fill_value=0.0)
            p_table.reset_index(inplace=True)
            """
            # print(p_table)

            # add first column as file_name
            summary_df = pd.concat([pd.Series([f[:3]] * len(ctry_arr), name="file_idx"), p_table], axis=1)
            print(summary_df)

            if read_mode == '1':
                aggr_df1 = pd.concat([aggr_df1, summary_df])
                # aggr_df1.reset_index(inplace=True)
            else:
                aggr_df2 = pd.concat([aggr_df2, summary_df])
                # aggr_df2.reset_index(inplace=True)

    if read_mode == '1':
        aggr_df1.to_csv(experi_path+"summary_1.csv")
    else:
        aggr_df2.to_csv(experi_path+"summary_2.csv")
