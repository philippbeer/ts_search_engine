from collections import namedtuple
from datetime import datetime
import math
from multiprocessing import cpu_count, Pool
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.seasonal import seasonal_decompose

import config

def read_csv(data: str)-> pd.DataFrame:
    """
    reads csv to df
    """
    return pd.read_csv(data)

def read_data() -> None:
    start_time = datetime.now()
    no_prc = cpu_count()-1

    # read csv files
    print("starting to read csv")
    df_l = []
    with Pool(processes=no_prc) as pool:
        for res in tqdm(pool.imap_unordered(read_csv, config.M4_FP_L),
                        total=len(config.M4_FP_L)):
            df_l.append(res)

    print("CSVs read after: {}".format(datetime.now()-start_time))

    print("concatenating dfs")
    df = pd.concat(df_l)
    return df

def get_trend(ts_ar: np.ndarray,
              periodicity: int) -> np.ndarray:
    if ts_ar.shape[0] < 2*periodicity:
        periodicity = math.floor(ts_ar.shape[0]/2)
        # raise Exception("ts_ar shape is: {}\nperiodicity is: {}".format(ts_ar.shape[0], periodicity))
    res = seasonal_decompose(ts_ar, 'additive', period=periodicity)
    # print("seas. decomp.: {}".format(res.trend))
    return res.trend[~np.isnan(res.trend)]

def fit_trend(trend_ar: np.ndarray) -> Tuple[float,float]:
    fit_res = np.polyfit(np.arange(trend_ar.shape[0]), trend_ar, 1)
    m = fit_res[0]
    b = fit_res[1]
    return (m,b)

def get_trend_coefs(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    ts_name = df['ts_name']
    ts_type = ts_name[0]
    if ts_type in config.PERIOD_MAPPING:
        period = config.PERIOD_MAPPING[ts_type]
    else:
        raise Exception("Period type not found")

    # get the reference data
    if "ref_data"in kwargs:
        df_all = kwargs['ref_data']
    # select time series from reference data
    df_ts = df_all[df_all['V1']==ts_name].iloc[:,1:].dropna(axis=1)
    df_stats = df_ts.transpose().describe()
    df_stats.columns = ['kpi']

    count = df_stats.loc['count', 'kpi']
    mean = df_stats.loc['mean', 'kpi']
    std = df_stats.loc['std', 'kpi']
    min_val = df_stats.loc['min', 'kpi']
    q25 = df_stats.loc['25%', 'kpi']
    q50 = df_stats.loc['50%', 'kpi']
    q75 = df_stats.loc['75%', 'kpi']
    max_val = df_stats.loc['max', 'kpi']
    
    # seasonal decomposition
    ar_ts = np.array(df_ts)[0]
    trend_ar = get_trend(ar_ts, period)
    m, b = fit_trend(trend_ar)

    # create named tuple
    TS_Result = namedtuple('TS_Result', 'm b count mean std min q25 q50 q75 max')
    res = TS_Result(m, b, count, mean, std, min_val,\
                    q25, q50, q75, max_val)
    return res


def main():
    tqdm.pandas()
    # read reference data
    df_ts = read_data()
    print("data read")

    # read top frequency data
    df_top = pd.read_csv("../data/df_freq_l.csv")
    df_top = df_top[:100]
    
    # res = df_top.apply(get_trend_coefs, 1, ref_data=df_ts)
    res = df_top.progress_apply(func=get_trend_coefs, axis=1, ref_data=df_ts)
    df_stats = pd.DataFrame(list(res))
    df_res = pd.merge(df_top, df_stats, left_index=True, right_index=True)
    
    df_res.to_csv("../data/df_stats.csv", index=False)

    print(df_res.head())
    print("\nstatistics and trend fit created")

    
if __name__ == "__main__":
    main()
