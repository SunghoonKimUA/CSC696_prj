import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

from datetime import datetime
from dateutil.relativedelta import relativedelta

from os import listdir
from os.path import isfile, join

def get_imf_val(df, ctry_cd, col_name):
    imf_val = np.nan
    if col_name in df.columns:
        tmp_df = df[df['Country Code'] == ctry_cd][col_name]
        tmp_df = tmp_df.dropna()
        if len(tmp_df) == 1:
            imf_val = float(tmp_df)
    return imf_val

def get_imf_gap(indicator, strt_val, end_val):
    imf_gap = 0.0
    if math.isnan(strt_val) or math.isnan(end_val):
        imf_gap = 0.0
    else:
        if '_IX' in indicator or '_PT' in indicator:
            #print(strt_prd_val, end_prd_val, end_prd_val-strt_prd_val)
            imf_gap = end_val-strt_val
        else:
            #print(strt_prd_val, end_prd_val, 99999.0 if strt_prd_val == 0 else np.sign(strt_prd_val)*round((end_prd_val-strt_prd_val)/strt_prd_val*100,3))
            if strt_val == 0.0 and end_val == 0.0:
                imf_gap = 0.0
            else:
                imf_gap = 99999.0 if strt_val == 0 else np.sign(strt_val)*round((end_val-strt_val)/strt_val*100,3)
    return imf_gap

# datasource : https://data.imf.org/ -> download imf data
# 1. declare each csv files and indicators from each csv files
imf_csv_indct = {'BOP':['`Indicator Code` == "I_BP6_USD"']
          ,'FM':['`Indicator Code` == "GGXCNL_G01_GDP_PT"']
          ,'FSI':['`Indicator Code` == "FSREPRR_PC_CP_A_PT"']
          ,'GFSMAB':['`Classification Code` == "GNLB|_Z" & `Sector Code` == "S13" & `Unit Code` == "XDC_R_B1GQ"'
                    ,'`Classification Code` == "GNOB|_Z" & `Sector Code` == "S13" & `Unit Code` == "XDC_R_B1GQ"']
          ,'IFS':['`Indicator Code` == "BK_DB_BP6_USD"'
                 ,'`Indicator Code` == "AIP_IX"'
                 ,'`Indicator Code` == "ENEER_IX"'
                 ,'`Indicator Code` == "NX_XDC"'
                 ,'`Indicator Code` == "FIMM_PA"'
                 ,'`Indicator Code` == "NGDP_XDC"'
                 ,'`Indicator Code` == "NM_XDC"'
                 ,'`Indicator Code` == "LE_IX"'
                 ,'`Indicator Code` == "ARS_IX"'
                 ,'`Indicator Code` == "FPE_IX"'
                 ,'`Indicator Code` == "IAFR_BP6_USD"'
                 ,'`Indicator Code` == "ILFR_BP6_USD"'
                 ,'`Indicator Code` == "PCPI_IX"']
          ,'IRFCL':['`Indicator Code` == "RAF_USD"']}
imf_csv_strt_idx = {'BOP':5, 'FM':5, 'FSI':5, 'GFSMAB':9, 'IFS':5, 'IRFCL':7}

# 2. read csv files into array(2nd column[ctry_code] and columns after imf_csv_strt_idx)
code_map_df = pd.read_csv("./data/imf_iso_ctrycode_map.csv", index_col=0, dtype={'imf':object})
code_map_dict = code_map_df.to_dict()["iso"]

imf_data_list = []

data_path = './data/imf/'
for f in listdir(data_path):
    if isfile(join(data_path, f)) and f.endswith('csv'):
        f_header = f.split('_',2)[0]
        data = pd.read_csv(join(data_path, f), low_memory=False, dtype={'Country Code':object})

        for condition in imf_csv_indct[f_header]:
            # print(condition)
            data1 = data.query(condition+" & Attribute == 'Value'").copy()
            drop_cols = []
            for drop_idx in range(imf_csv_strt_idx[f_header]):
                # col_idx = 1 means ctry_cd
                if drop_idx != 1:
                    drop_cols.append(drop_idx)
            # print(data1.columns[drop_cols])
            data1.drop(data1.columns[drop_cols], axis=1, inplace=True)
            # convert imf_numeric_country_code into iso_char_country_code
            data1 = data1.replace({'Country Code':code_map_dict})
            # remain only converted_country_code
            data1 = data1[data1['Country Code'].str.match("^[A-Z]+")]

            imf_data_list.append({condition:data1})

# 3. build new csv with below columns
#    iso_code, oecd_eval_dt, list of values in economic indicators
#    empty values is filled with 0.0
imf_df_list = []
imf_df_columns = []

# print(imf_data_list)

# date initialisation
S = pd.read_csv("./data/cre-crc-historical-internet-english.csv", index_col=0)
end_dt_lst = S.columns[3:]

# col2
for dt in end_dt_lst:
    print(dt)
    dt_str = dt
    dt = datetime.strptime(dt,"%d-%b-%y")
    dt_prev_mon = dt - relativedelta(months=1)
    dt_prev_qt = dt - relativedelta(months=3)
    #print(dt.year)
    # col1
    for ctry_cd in code_map_df['iso']:
        ctry_idc_val = [ctry_cd, dt_str]
        
        if ctry_cd == 'USA':
            imf_df_columns = ['ctry_cd', 'dt_dtr']
        
        for each_data_list in imf_data_list:
            for k, v in each_data_list.items():
                if ctry_cd == 'USA':
                    print(k.split("==")[1].strip())
                    print(imf_df_columns)
                    imf_df_columns.append(k.split("==")[1].strip().replace('"',''))
                # print(k)
                # print(ctry_cd)
                if len(v[v['Country Code'] == ctry_cd]) == 0:
                    ctry_idc_val.append(0.0)
                else:
                    # print(str(dt_prev_mon.year-1)+'M'+str(dt_prev_mon.month), str(dt_prev_qt.year-1)+'Q'+str(int((dt_prev_qt.month-1)/3)+1))
                    strt_m_val = get_imf_val(v, ctry_cd, str(dt_prev_mon.year-1)+'M'+str(dt_prev_mon.month))
                    end_m_val = get_imf_val(v, ctry_cd, str(dt_prev_mon.year-0)+'M'+str(dt_prev_mon.month))
                    strt_q_val = get_imf_val(v, ctry_cd, str(dt_prev_qt.year-1)+'Q'+str(int((dt_prev_qt.month-1)/3)+1))
                    end_q_val = get_imf_val(v, ctry_cd, str(dt_prev_qt.year-0)+'Q'+str(int((dt_prev_qt.month-1)/3)+1))
                    strt_y_val = get_imf_val(v, ctry_cd, str(dt.year-2))
                    end_y_val = get_imf_val(v, ctry_cd, str(dt.year-1))

                    #if ctry_cd == 'AUS' and 'BK_DB_BP6_USD' in k:
                    #    print(math.isfinite(strt_m_val), strt_q_val, end_q_val, math.isfinite(strt_y_val), str(dt_prev_qt.year-1)+'Q'+str(int((dt_prev_qt.month-1)/3)+1))

                    # step 1. get monthly value
                    if math.isfinite(strt_m_val) and math.isfinite(end_m_val) and strt_m_val != 0.0 and end_m_val != 0.0:
                        ctry_idc_val.append(get_imf_gap(k, strt_m_val, end_m_val))
                    # step 2. get quaterly value
                    elif math.isfinite(strt_q_val) and math.isfinite(end_q_val) and strt_q_val != 0.0 and end_q_val != 0.0:
                        ctry_idc_val.append(get_imf_gap(k, strt_q_val, end_q_val))
                    # step 3. get yearly value
                    else:
                        ctry_idc_val.append(get_imf_gap(k, strt_y_val, end_y_val))
        print(ctry_idc_val)
        imf_df_list.append(ctry_idc_val)
#                except:
#                    print("aaa",strt_prd_val,end_prd_val)
#                    raise Exception("Sorry, no numbers below zero")
#            imf_df_list.append()

imf_df = pd.DataFrame(imf_df_list)
imf_df.to_csv('./data/imf_df.csv',header=imf_df_columns)

