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
    global df_g_fft
    global df_g_ham
    global df_g_welch
    df_g_fft = df[df['type']=='fft']
    df_g_ham = df[df['type']=='Hamming']
    df_g_welch = df[df['type']=='Welch']

def extend_missing_freqs(freqs_l: list) -> list:
    # missing frequencies indicator = -1
    
    freqs_ext = [-1]*(5-len(freqs_l))
    freqs_ext.extend(freqs_l)
    
    return freqs_ext

def compare_freqs(df_sub: pd.DataFrame,
                  freq_l: list) -> pd.Series:
    # convert frequencies column to array
    candidates = np.array(list(df_sub['freq_ids']))
    template_ar = np.array([freq_l]*candidates.shape[0])
    # match the frequencies
    match_ar =  candidates-template_ar 
    mask = match_ar == 0 
    indx = np.nonzero(mask)

    """
    on the matches apply 10 to power of column index
    and set non-matches to zero
    """
    match_ar[mask] = 10**(indx[1])
    match_ar[~mask] = 0

    match_scores = pd.Series(np.sum(match_ar, axis=1))
    return match_scores

def get_global_df(fft_type: str) -> pd.DataFrame:
    # select correct global df
    if fft_type == config.FFT:
        df_sub = df_g_fft
    elif fft_type == config.HAM:
        df_sub = df_g_ham
    elif fft_type == config.WELCH:
        df_sub = df_g_welch
    else:
        raise Exception('unidentified window type')
    return df_sub

def generate_stats(stats_ar: np.ndarray) -> pd.DataFrame:
    """
    compute match coefficient for each column
    """
    ts_1 = stats_ar[0]
    freq_l = stats_ar[1]
    fft_type = stats_ar[2]
    m = stats_ar[3]
    # b = ts_stats_ar[4]
    count = stats_ar[5]
    mean = stats_ar[6]
    std = stats_ar[7]
    # min = ts_stats_ar[8]
    q25 = stats_ar[9]
    q50 = stats_ar[10]
    q75 = stats_ar[11]
    # max = ts_stats_ar[12]
    
    # ts_1 = s['ts_name']
    # freq_l = s['freq_ids']
    # fft_type = s['type']
    # mean = s['mean']
    # m = s['m']
    # count = s['count']
    # std = s['std']
    # q25 = s['q25']
    # q50 = s['q50']
    # q75 = s['q75']
    

    df_sub = get_global_df(fft_type)
    # remove unnecessary candidates from df_g
    if m>0:
        df_sub = df_sub[(df_sub['m']>0)]
    elif m==0:
        raise Exception("handle special case of exact stationarity")
    else:
        df_sub = df_sub[(df_sub['m']<0)]

    # create result df
    cols = ['ts_1', 'ts_2', 'type', 'match_score', 'd_m',
            'd_mean', 'd_std', 'd_count', 'd_q25', 'd_q50', 'd_q75']
    df_res = pd.DataFrame(columns=cols)
    df_res['ts_1'] = [ts_1]*df_sub.shape[0]
    df_res['ts_2'] = df_sub['ts_name'].reset_index(drop=True)
    df_res['type'] = [fft_type]*df_sub.shape[0]

    # compare frequencies
    match_scores = compare_freqs(df_sub, freq_l)
    if match_scores.shape[0] == df_sub.shape[0]:
        df_res['match_score'] = match_scores
    else:
        raise Exception("match scores must match dataframe")

    # compute delta statistics
    df_res['d_m'] = df_sub['m'].apply(lambda x: abs(x-m)).reset_index(drop=True)
    df_res['d_mean'] = df_sub['mean'].apply(lambda x: abs(x-mean)).reset_index(drop=True)
    df_res['d_std'] = df_sub['std'].apply(lambda x: abs(x-std)).reset_index(drop=True)
    df_res['d_count'] = df_sub['count'].apply(lambda x: abs(x-mean)).reset_index(drop=True)
    df_res['d_q25'] = df_sub['q25'].apply(lambda x: abs(x-q25)).reset_index(drop=True)
    df_res['d_q50'] = df_sub['q50'].apply(lambda x: abs(x-q50)).reset_index(drop=True)
    df_res['d_q75'] = df_sub['q75'].apply(lambda x: abs(x-q75)).reset_index(drop=True)

    # drop where time series match
    df_res = df_res[~(df_res['ts_1']==df_res['ts_2'])]
    return df_res
    
    
def main():
    start_time = datetime.now()
    no_cpu = cpu_count()-1
    df = pd.read_csv(config.TS_STATS_FP,
                     converters={'freq_ids': lambda x: eval(x)})
    # df = df.sample(100)
    print("df with shape {} read".format(df.shape))
    print("data loading finished after {}s".format(datetime.now()-start_time))
    # add missing frequencies place holders in case they exist
    df['freq_ids'] = df['freq_ids'].apply(extend_missing_freqs)

    # set_df(df) # set global variables

    # tqdm.pandas()
    # print("starting apply")
    # res  = df.progress_apply(func=generate_stats, axis=1)
    # df_res = pd.concat(list(res))

    # # ensuring order of columns
    # df_analyze = df[(df['ts_name'].str.contains('H') |\
    #                  (df['ts_name'].str.contains('D')))]
    df = df[['ts_name',	'freq_ids', 'type', 'm', 'b', 'count', 'mean', 'std'\
             , 'min', 'q25', 'q50', 'q75', 'max']]
    df_analyze = df.sample(1000)
    # # generating list to be processed via multiprocessing
    ts_l = list(df_analyze.values)
    
    res_l = []  # results list
    print("starting multiprocessing")
    with Pool(processes=no_cpu,
              initializer=set_df,
              initargs=(df,)) as pool:
        for res in tqdm(pool.imap_unordered(generate_stats,ts_l,
                                            chunksize=50),
                        total=len(ts_l)):
            # df_tmp = pd.concat(list(res))
            res_l.append(res)

    df_res = pd.concat(res_l)


    df_res.to_csv("../data/df_match_ts_1000.csv", index=False)
    print("data written to file")
    print("completed in {}".format(datetime.now()-start_time))

if __name__ == "__main__":
    main()
