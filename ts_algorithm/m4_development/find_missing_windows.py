"""
This program finds the closest series based on the defined criteria. This matches the series and finds the closest
"""
from datetime import datetime
import timeit
from multiprocessing import Pool
from os import cpu_count
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cnf

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
    if fft_type == cnf.FFT:
        df_sub = df_g_fft
    elif fft_type == cnf.HAM:
        df_sub = df_g_ham
    elif fft_type == cnf.WELCH:
        df_sub = df_g_welch
    else:
        raise Exception('unidentified window type')
    return df_sub

def generate_stats(stats_ar: np.ndarray) -> pd.DataFrame:
    """
    compute match coefficient for each column
    """
    ts_1 = stats_ar[0]
    no_1 = stats_ar[1]
    cls_type = stats_ar[2]
    freq_l = stats_ar[3]
    fft_type = stats_ar[4]
    m = stats_ar[5]
    b = stats_ar[6]
    count = stats_ar[7]
    mean = stats_ar[8]
    std = stats_ar[9]
    min_val = stats_ar[10]
    q25 = stats_ar[11]
    q50 = stats_ar[12]
    q75 = stats_ar[13]
    max_val = stats_ar[14]

    df_sub = get_global_df(fft_type)
    # remove unnecessary candidates from df_g
    if m>0:
        df_sub = df_sub[(df_sub['m']>0)]
    elif m==0:
        raise Exception("handle special case of exact stationarity")
    else:
        df_sub = df_sub[(df_sub['m']<0)]

    cols = ['ts_1', 'no_1', 'ts_2', 'no_2', 'type', 'match_score', 'd_m',
            'd_mean', 'd_std', 'd_count', 'd_q25', 'd_q50', 'd_q75']
    df_res = pd.DataFrame(columns=cols)
    df_res['ts_1'] = [ts_1]*df_sub.shape[0]
    df_res['no_1'] = no_1
    df_res['class_1'] = cls_type
    df_res['ts_2'] = df_sub['ts_name'].reset_index(drop=True)
    df_res['no_2'] = df_sub['no'].reset_index(drop=True)
    df_res['class_2'] = df_sub['class'].reset_index(drop=True)
    df_res['type'] = [fft_type]*df_sub.shape[0]

    match_scores = compare_freqs(df_sub, freq_l)
    if match_scores.shape[0] == df_sub.shape[0]:
        df_res['match_score'] = match_scores
    else:
        df_res['match_score'] = match_scores

    # compute delta statistics
    df_res['d_m'] = df_sub['m'].apply(lambda x: abs(x-m)).reset_index(drop=True)
    df_res['d_mean'] = df_sub['mean'].apply(lambda x: abs(x-mean)).reset_index(drop=True)
    df_res['d_std'] = df_sub['std'].apply(lambda x: abs(x-std)).reset_index(drop=True)
    df_res['d_count'] = df_sub['count'].apply(lambda x: abs(x-count)).reset_index(drop=True)
    df_res['d_q25'] = df_sub['q25'].apply(lambda x: abs(x-q25)).reset_index(drop=True)
    df_res['d_q50'] = df_sub['q50'].apply(lambda x: abs(x-q50)).reset_index(drop=True)
    df_res['d_q75'] = df_sub['q75'].apply(lambda x: abs(x-q75)).reset_index(drop=True)
    df_res['d_min'] = df_sub['min'].apply(lambda x: abs(x-min_val)).reset_index(drop=True)
    df_res['d_max'] = df_sub['max'].apply(lambda x: abs(x-max_val)).reset_index(drop=True)
    
    # drop where time series match
    # df_res = df_res[~((df_res['ts_1']==df_res['ts_2']) &
    #                   (df_res['no_1']==df_res['no_2']))]

    # drop frequency matches below 10^4
    if df_res.shape[0] == 0:
        print(f"{ts_1} - {no_1} - {fft_type}")

    df_res = df_res[df_res['match_score']>cnf.MATCH_SCORE_THRESH]
    return df_res

def read_ucr_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    read ucr converted time series for test and train set
    """
    df_train = pd.read_csv(cnf.UCR_TS_STATS_TRAIN_FP,
                           converters = {"freq_ids": lambda x: eval(x)})
    df_test = pd.read_csv(cnf.UCR_TS_STATS_TEST_FP,
                          converters = {"freq_ids": lambda x: eval(x)})
    
    df_train['no'] = df_train['no'].astype(int)
    df_train['class'] = df_train['no'].astype(int)
    df_test['no'] = df_test['no'].astype(int)
    
    # add missing frequencies
    df_train['freq_ids'] = df_train['freq_ids'].apply(extend_missing_freqs)
    df_test['freq_ids'] = df_test['freq_ids'].apply(extend_missing_freqs)
    
    return df_train, df_test


def main():
    start_time = datetime.now()
    no_cpu = cpu_count()-1

    df_train, df_test = read_ucr_data()

    ##############
    print("data loading finished after {}s".format(datetime.now()-start_time))

    df_train = df_train[['ts_name', 'no', 'class', 'freq_ids', 'type', 'm'\
                         , 'b', 'count', 'mean', 'std', 'min', 'q25'\
                         , 'q50', 'q75', 'max']]

    df_test = df_test[['ts_name', 'no', 'class', 'freq_ids', 'type', 'm'\
                         , 'b', 'count', 'mean', 'std', 'min', 'q25'\
                         , 'q50', 'q75', 'max']]


    # filtering on window type to avoid duplicates due to window type
    # df_sample = df_test[df_test['type']=='fft'].groupby('ts_name')\
    #     ['ts_name', 'no'].apply(lambda x: x.sample(min(len(x), 10)))
    # filter = list(zip(df_sample['ts_name'], df_sample['no'])) # extracting the key combinations to filter

    #rerun with reference frequency
    # df_dtw = pd.read_csv("../data/df_dtw_comp.csv")
    df_missing = pd.read_csv("../data/missing_windows.csv")
    filter = list(zip(df_missing['ts_1'],df_missing['no_1']))
    # ensure that each window type is available for sampled time series
    df_sample = df_test[df_test.set_index(['ts_name', 'no']).index.isin(filter)]
    ts_l = list(df_sample.values) # convert dataframe to list of elements

    res_l = []  # results list
    print("starting multiprocessing")
    with Pool(processes=no_cpu,
              initializer=set_df,
              initargs=(df_train,)) as pool:
        for res in tqdm(pool.imap_unordered(generate_stats,ts_l,
                                            chunksize=50),
                        total=len(ts_l)):
            res_l.append(res)

    df_res = pd.concat(res_l)


    df_res.to_csv("../data/res_missing_wdw.csv", index=False)
    print("data written to file")
    print("completed in {}".format(datetime.now()-start_time))

if __name__ == "__main__":
    main()
