from  multiprocessing import Pool, cpu_count
import os
import random
from typing import Tuple, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

# loading results

def write_f_to_cols(data: Tuple[str,pd.DataFrame]) -> pd.DataFrame:
    N = 5
    ts_name = data[0]
    df = data[1]

    
    df['freq_approx_idx'] = df['freq_approx_idx'].astype(int)

    freq_l = df['freq_approx_idx'].tolist()

    idx_largest_freq = sorted(range(len(freq_l)), key= lambda x: freq_l[x])[-N:]

    df_res = pd.DataFrame({'ts_name': ts_name,
                           'freq_ids': freq_l})
    return df_res

def compare_frequencies(data: Tuple[str,Set[int]]) -> pd.DataFrame:
    ts_name = data[0]
    fappr_idx_s = data[1]
    f_no = len(fappr_idx_s)
    ts_1 = []
    ts_2 = []
    f_intersec = []

    df_sub = df[df['ts_name']!=ts_name]
    for ts_sub in df_sub.groupby('ts_name'):
        ts_name_sub = ts_sub[0]
        ts_df_sub = ts_sub[1]
        # print("f_no: {}".format(f_no))
        # print("shape: {}".format(ts_df_sub.shape))
        if (f_no/ts_df_sub.shape[0]) > 1:
            q = 1
        else:
            q = (f_no/ts_df_sub.shape[0])
        thresh_sub = np.quantile(ts_df_sub['PSD'], q)
        core_df_sub = ts_df_sub[ts_df_sub['PSD']>thresh_sub]
        fappr_idx_sub_s = set(core_df_sub['freq_approx_idx'])
        no_intersection = len(fappr_idx_s & fappr_idx_sub_s)
        ts_1.append(ts_name)
        ts_2.append(ts_name_sub)
        f_intersec.append(no_intersection)
    data = {'ts_1': ts_1,
            'ts_2': ts_2,
            'f_intersec':f_intersec}
    df_f_sub = pd.DataFrame(data)
    df_f_sub.sort_values(by="f_intersec", ascending=False,
                        inplace=True)
    return df_f_sub.iloc[:50,:] # keep top 50 rows        

    
def main():
    no_prc = cpu_count()-1
    print("starting computation of top 5 frequencies per ts")
    df = pd.read_csv("../data/df_apx_win.csv")
    print("df read with shape: {}".format(df.shape))

    ts_l = []
    df_res = pd.DataFrame(columns=['ts_name', 'freq_ids'])
    for ts in tqdm(df.groupby(["type", "ts_name"])):
        print()
        # ts_l.append(ts)

        N = 5
        ts_type = ts[0][0]
        ts_name = ts[0][1]
        df_sub = ts[1]

    
        df_sub['freq_apx_idx'] = df_sub['freq_apx_idx'].astype(int)

        freq_l = df_sub['freq_apx_idx'].tolist()
        PSD_l = df_sub['PSD'].tolist()

        idx_powerful_PSD = sorted(range(len(PSD_l)), key= lambda
                                  x: PSD_l[x])[-N:]

    
        # get the largest values by their index into the list
        freq_idx = [freq_l[i] for i in idx_powerful_PSD]
    
        df_res = df_res.append(pd.DataFrame({'ts_name': ts_name,
                                             'type': ts_type,
                                             'freq_ids': str(freq_idx)},
                                            index=[0]))
 
        
    df_res.reset_index(inplace=False)
    df_res.to_csv("../data/df_freq_l.csv", index=False)



    main()
