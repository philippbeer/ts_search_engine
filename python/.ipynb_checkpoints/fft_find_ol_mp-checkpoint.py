from multiprocessing import Pool
from os import cpu_count

from typing import Tuple

import pandas as pd
from tqdm import tqdm

def compute_match(ts: pd.Series) -> pd.DataFrame:
    ts_1 = []
    ts_2 = []
    scores = []
    for k, r in df_g.iterrows():
        if r['ts_name']==ts['ts_name']:
            continue
        else:
            l_tmp = [10**i if ts['freq_ids'][i]==r['freq_ids'][i]\
                 else 0 for i in range(len(r['freq_ids']))]
            match = sum(l_tmp)
            ts_1.append(ts['ts_name'])
            ts_2.append(r['ts_name'])
            scores.append(match)

    df_res = pd.DataFrame({'ts_1': ts_1,
                           'ts_2': ts_2,
                           'match_score': scores})
    df_res.sort_values('match_score', ascending=False,
                       inplace=True)
    df_res.reset_index(inplace=True)
    return df_res.iloc[:15]

def set_df(df: pd.DataFrame):
    global df_g
    df_g = df

def main():
    no_cpu = cpu_count()-1
    df = pd.read_csv("df_freq_l.csv",
                     converters={'freq_ids': lambda x: eval(x)})
    # df = df.sample(10000)
    df = df[df['ts_name'].str.contains('H')]
    print("df with shape {} read".format(df.shape))
    df_res = pd.DataFrame(columns=['ts_1', 'ts_2', 'matching_score'])
    ts_l = []
    for k, r in tqdm(df.iterrows(), total=\
                     df.shape[0]):
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


    df_res.to_csv("df_matching_scores.csv")
            

if __name__ == "__main__":
    main()
