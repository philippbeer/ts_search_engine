"""
This program finds the closest series based on the defined criteria. This matches the series and finds the closest
"""
from datetime import datetime
from multiprocessing import Pool
from os import cpu_count
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import config

def compute_match(ts_stats_ar: np.ndarray) -> pd.DataFrame:
    ts_1 = []
    ts_2 = []
    fft_type = []
    delta_m = []
    delta_mean = []
    delta_q25 = []
    delta_median = []
    delta_q75 = []
    delta_std = []
    delta_count = []
    scores = []

    # assign array values to corresponding variables
    ts_name = ts_stats_ar[0]
    freq_ids = ts_stats_ar[1]
    ts_type = ts_stats_ar[2]
    m = ts_stats_ar[3]
    # b = ts_stats_ar[4]
    count = ts_stats_ar[5]
    mean = ts_stats_ar[6]
    std = ts_stats_ar[7]
    # min = ts_stats_ar[8]
    q25 = ts_stats_ar[9]
    q50 = ts_stats_ar[10]
    q75 = ts_stats_ar[11]
    # max = ts_stats_ar[12]

    # reduce all dataframes to ones that match the trend direction
    # only compare ts created by same type of transformation
    if m>0:
        df_sub = df_g[(df_g['m']>0)\
                      & (df_g['type']==ts_type)\
                      & ((df_g['mean']<=config.MEAN_THRESH*mean)\
                         & (df_g['mean']>=(mean-config.MEAN_THRESH*mean)))]
    elif m==0:
        raise Exception("handle special case of exact stationarity")
    else:
        df_sub = df_g[(df_g['m']<0)\
                      & (df_g['type']==ts_type)\
                      & ((df_g['mean']<=config.MEAN_THRESH*mean)\
                         & (df_g['mean']>=mean-config.MEAN_THRESH*mean))]


    for k, r in df_sub.iterrows():
        if r['ts_name']==ts_name:
            continue
        else:
            l_tmp = [10**i if freq_ids[i]==r['freq_ids'][i]\
                 else 0 for i in range(len(r['freq_ids']))]

            match = sum(l_tmp)
            ts_1.append(ts_name)
            ts_2.append(r['ts_name'])
            fft_type.append(ts_type)
            delta_m.append(abs(r['m']-m))
            delta_mean.append(abs(r['mean']-mean))
            delta_q25.append(abs(r['q25']-q25))
            delta_median.append(abs(r['q50']-q50))
            delta_q75.append(abs(r['q75']-q75))
            delta_std.append(abs(r['std']-std))
            delta_count.append(abs(r['count']-count))
            scores.append(match)
    
    
    df_res = pd.DataFrame({'ts_1': ts_1,
                           'ts_2': ts_2,
                           'type': fft_type,
                           'match_score': scores,
                           'd_m': delta_m,
                           'd_mean': delta_mean,
                           'd_q25': delta_q25,
                           'd_q50': delta_median,
                           'd_q75': delta_q75,
                           'd_std': delta_std,
                           'd_count': delta_count})
    sort_cols = ['match_score', 'd_m', 'd_mean', 'd_std', 'd_count',\
                 'd_q50', 'd_q75', 'd_q25']
    sort_order = [False,True,True,True,True,\
                  True,True,True]
    df_res.sort_values(sort_cols, ascending=sort_order,
                       inplace=True)
    df_res.reset_index(inplace=True)
    return df_res

def set_df(df: pd.DataFrame):
    global df_g
    df_g = df

def extend_missing_freqs(freqs_l: list) -> list:
    # missing frequencies indicator = -1
    freqs_ext = [-1]*(5-len(freqs_l))
    freqs_ext.extend(freqs_l)
    return freqs_ext

    
def main():
    start_time = datetime.now()
    no_cpu = cpu_count()-1
    df = pd.read_csv(config.TS_STATS_FP,
                     converters={'freq_ids': lambda x: eval(x)})
    # df = df.sample(10)
    print("df with shape {} read".format(df.shape))
    print("data loading finished after {}s".format(datetime.now()-start_time))
    # add missing frequencies place holders in case they exist
    df['freq_ids'] = df['freq_ids'].apply(extend_missing_freqs)

    # ensuring order of columns
    df_analyze = df[(df['ts_name'].str.contains('H') |\
                     (df['ts_name'].str.contains('D')))]
    df_analyze = df_analyze[['ts_name',	'freq_ids', 'type', 'm', 'b', 'count', 'mean', 'std'\
             , 'min', 'q25', 'q50', 'q75', 'max']]

    # generating list to be processed via multiprocessing
    ts_l = list(df_analyze.values)
    
    res_l = []  # results list
    print("starting multiprocessing")
    with Pool(processes=no_cpu,
              initializer=set_df,
              initargs=(df,)) as pool:
        for res in tqdm(pool.imap_unordered(compute_match,ts_l,
                                            chunksize=50),
                        total=len(ts_l)):
            res_l.append(res)

    df_res = pd.concat(res_l)


    df_res.to_csv("../data/df_match_ts.csv")
    print("data written to file")
    print("completed in {}".format(datetime.now()-start_time))

if __name__ == "__main__":
    main()
