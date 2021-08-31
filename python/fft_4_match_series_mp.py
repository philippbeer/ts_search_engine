"""
This program finds the closest series based on the defined criteria. This matches the series and finds the closest
"""

from multiprocessing import Pool
from os import cpu_count
from typing import Tuple

import pandas as pd
from tqdm import tqdm

import config

def compute_match(ts: pd.Series) -> pd.DataFrame:
    ts_1 = []
    ts_2 = []
    delta_m = []
    delta_mean = []
    delta_q25 = []
    delta_median = []
    delta_q75 = []
    delta_std = []
    delta_count = []
    scores = []

    # reduce all dataframes to ones that match the trend direction
    if ts['m']>0:
        df_sub = df_g[df_g['m']>0]
    elif ts['m']==0:
        raise Exception("handle special case of exact stationarity")
    else:
        df_sub = df_g[df_g['m']<0]

    print("ts freq length: {}".format(len(ts['freq_ids'])))
    for k, r in df_sub.iterrows():
        if r['ts_name']==ts['ts_name']:
            continue
        else:
            if len(ts['freq_ids'])!=len(r['freq_ids']):
                print("clash - length {}\nts: {}\nr: {}".format(len(r['freq_ids']), ts['ts_name'], r['ts_name']))
            l_tmp = [10**i if ts['freq_ids'][i]==r['freq_ids'][i]\
                 else 0 for i in range(len(r['freq_ids']))]
            match = sum(l_tmp)
            ts_1.append(ts['ts_name'])
            ts_2.append(r['ts_name'])
            delta_m.append(abs(r['m']-ts['m']))
            delta_mean.append(abs(r['mean']-ts['mean']))
            delta_q25.append(abs(r['q25']-ts['q25']))
            delta_median.append(abs(r['q50']-ts['q50']))
            delta_q75.append(abs(r['q75']-ts['q75']))

            delta_count.append(abs(r['count']-ts['count']))
            scores.append(match)

    print("########### Length of lists ##########")
    print("ts: {}".format(ts['ts_name']))
    if len(ts_1) == len(ts_2) == len(scores)\
       == len(delta_m) == len(delta_mean) == len(delta_q25)\
       == len(delta_median) == len(delta_q75):
        pass
    else:
        print("issue with round: {}\nseries: {}\nchecking against: {}".format(k, r['ts_name'], ts['name']))
    df_res = pd.DataFrame({'ts_1': ts_1,
                           'ts_2': ts_2,
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
    print("series {} completed".format(ts['ts_name']))
    return df_res

def set_df(df: pd.DataFrame):
    global df_g
    df_g = df

def main():
    no_cpu = cpu_count()-1
    df = pd.read_csv(config.TS_STATS_FP,
                     converters={'freq_ids': lambda x: eval(x)})
    # df = df.sample(1000)
    df_analyze = df[(df['ts_name'].str.contains('H')) | (df['ts_name'].str.contains('D'))]
    print("df with shape {} read".format(df.shape))
    print(df.head())
    df_res = pd.DataFrame(columns=['ts_1', 'ts_2', 'matching_score'])
    ts_l = []
    for k, r in tqdm(df_analyze.iterrows(), total=len(ts_l)):
        ts_l.append(r)

    res_l = []
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

if __name__ == "__main__":
    main()
