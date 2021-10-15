import time
from math import floor
from multiprocessing import Pool, cpu_count
from typing import List

import pandas as pd
from tqdm import tqdm

import config as cnf

def set_df(df: pd.DataFrame) -> None:
    """
    setting the global dataframe
    """
    global df_g
    df_g = df

def find_delta_min(s: pd.Series) -> pd.Series:
    """
    compute the best matches for the series at hand
    """
    df_match = df_g[(df_g['ts_1']==s['ts_1'])\
                    & (df_g['no_1']==s['no_1'])\
                    & (df_g['type']==s['type'])]

    # reducing to highest matching frequencies
    max_f_match = df_match['match_score'].max()
    df_match = df_match[df_match['match_score']==max_f_match]

    df_res = pd.DataFrame()
    # get lowest delta matches per kpi
    for i in range(len(cnf.UCR_DELTA_KPI)):
        kpi = cnf.UCR_DELTA_KPI[i]
        min_series = df_match.sort_values(kpi).iloc[0,:]
        df_tmp = pd.DataFrame({
                        'ts_1': s['ts_1'],
            'no_1': s['no_1'],
            'class_1': s['class_1'],
            'ts_2': min_series['ts_2'],
            'no_2': min_series['no_2'],
            'class_2': min_series['class_2'],
            'type': min_series['type'],
            'match_score': max_f_match,
            'min_kpi': kpi,
            'value': min_series[kpi]
        }, index=[i])

        df_res = df_res.append(df_tmp)

    return df_res

def find_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    generate the dataframe containing the matches
    """
    df_res = pd.concat(list(df.apply(find_delta_min, 1)))
    
    return df_res

def split_df(df: pd.DataFrame,
             pieces: int = 7)->List[pd.DataFrame]:
    """
    splitting data frame to list of dataframes
    """
    entries = df.shape[0]
    step = floor(entries/pieces)
    start = 0
    stop = step

    df_l = []
    while(stop < entries):
        df_tmp = df[start:stop]
        df_l.append(df_tmp)
        
        start = stop
        stop = stop + step

    # getting the rest of the entries not captured by while loop
    df_tmp = df[start:]
    df_l.append(df_tmp)

    return df_l
    
    
def main():
    no_prc = cpu_count()-1
    start = time.time()
    print("reading file")
    df = pd.read_csv(cnf.UCR_MATCH_RES_FP)
    csv_read = time.time()
    print("file read in {}".format(csv_read - start))
    df['class_1'] = df['class_1'].astype(int)
    df['class_2'] = df['class_2'].astype(int)


    # get unique combinations
    print("creating unique combinations")
    df_comb = pd.DataFrame(list(df.set_index(['ts_1', 'no_1',\
                                              'class_1', 'type'])\
                                .index.unique()),
                           columns=['ts_1', 'no_1', 'class_1', 'type'])
    unique_comb = time.time()
    print("unique combinations created in {}".format(unique_comb-csv_read))
    df_l = split_df(df_comb)
    res_l = []

    print("start multiprocessing")
    with Pool(processes=no_prc,
              initializer=set_df,
              initargs=(df,)) as pool:
        for res in tqdm(pool.imap_unordered(find_matches, df_l),
                        total=len(df_l)):
            res_l.append(res)

    mp_time = time.time()
    print(" multiprocessing completed in {}".format(mp_time-unique_comb))

    print("concatenate dfs")
    df_res = pd.concat(res_l)
    df_res.sort_values(['ts_1', 'no_1', 'match_score', 'type'], inplace=True)
    concat_time = time.time()
    print("concatenated in {}".format(concat_time-mp_time))

    print("writing to file")
    df_res.to_csv(cnf.UCR_MIN_DELTA_FP, index=False)
    written_time = time.time()
    print("written to file in {}".format(written_time-concat_time))
    print("execution completed in: {}".format(time.time()-start))
    
if __name__ == "__main__":
    main()
