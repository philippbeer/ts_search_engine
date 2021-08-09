from multiprocessing import Pool, cpu_count
import os
from typing import Tuple, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

# loading results
df = pd.read_csv("df_approx.csv")
#df = df.iloc[:10000,:]

def write_f_to_cols(df: pd.DataFrame):
    df['freq_approx_idx'] = df['freq_approx_idx'].astype(str)

    freq_l = list(df['freq_approx_idx'])
    ts_name = df.at[0,0]

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

    ts_l = f[]
    for ts in tqdm(df.groupby("ts_name")):
        ts_l.append(ts)

    freq_df_l = []
    with Pool(processes=no_prc) as pool:
        for res in tqdm(pool.imap_unordered(write_f_to_cols, ts_l), total=len(ts_l)):
            freq_df_l.append(res)

    df_res = pd.concat(freq_df_l)

    df_res.to_csv("df_freq_l.csv", index=Fales)

    # q_thresh = 0.9
    # cols = ['ts_1', 'ts_2', 'f_intersec']
    # ts_l = []
    # for ts in tqdm(df.groupby("ts_name")):
    #     ts_name = ts[0]
    #     ts_df = ts[1]
        
    #     thresh = np.quantile(ts_df['PSD'], q_thresh)
    #     core_df = ts_df[ts_df['PSD'] > thresh]
    #     fappr_idx_s = set(core_df['freq_approx_idx'])
    #     f_no = len(fappr_idx_s)
    #     ts_l.append((ts_name,fappr_idx_s))

    # comp_df_l = []
    # with Pool(processes=no_prc) as pool:
    #     for res in tqdm(pool.imap_unordered(compare_frequencies, ts_l)):
    #         comp_df_l.append(res)

    # df_res = pd.concat(comp_df_l)
    # df_res.to_csv("df_f_overlap.csv", index=False)



if __name__ == "__main__":
    main()
