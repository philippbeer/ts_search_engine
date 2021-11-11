from typing import Tuple

import pandas as pd
from tqdm import tqdm

def compute_match(ts: pd.Series) -> pd.DataFrame:
    ts_2 = []
    scores = []
    for k, r in df.iterrows():
        l_tmp = [10**i if ts['freq_ids'][i]==r['freq_ids'][i]\
                 else 0 for in range(df.shape[0])]
        match = sum(l_tmp)
        ts_2.append(r['ts_name'])
        scores.append(match)

    df_res = pd.DataFrame({'ts_1': r['ts_name']*df.shape[0],
                           'ts_2': ts_2,
                           'match_score': scores})
    df_res.sort_values('match_score', ascending=False,
                       inplace=True)
    df_res.reset_index(inplace=True)
    return df_res

def main():
    df = pd.read_csv("df_freq_l.csv",
                     converters={'freq_ids': lambda x: eval(x)})

    print("df with shape {} read".format(df.shape))
    df_res = pd.DataFrame(columns=['ts_1', 'ts_2', 'matching_score'])
    for k, r in tqdm(df.iterrows(), total=
                     df.shape[0]):
        df_sub = df[df['ts_name']!=r['ts_name']]

        ts_1 = []
        ts_2 = []
        score = []
        for k_sub, r_sub in df_sub.iterrows():
            # print("type of row list: {}".format(r['freq_ids']))
            # print('type of row sub list: {}'.format(r_sub['freq_ids']))
            
            l_tmp = [10**i if r['freq_ids'][i] == r_sub['freq_ids'][i] else 0\
             for i in range(len(r_sub['freq_ids']))]
            match = sum(l_tmp)
            ts_1.append(r['ts_name'])
            ts_2.append(r_sub['ts_name'])
            score.append(match)

        df_tmp = pd.DataFrame({'ts_1': ts_1,
                              'ts_2': ts_2,
                              'matching_score': score})

        df_tmp.sort_values('matching_score', ascending=False,
                           inplace=True)
        df_res = df_res.append(df_tmp)

    df_res.to_csv("df_matching_scores.csv")
            

if __name__ == "__main__":
    main()
