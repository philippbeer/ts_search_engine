import argparse
from datetime import datetime
import math
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm

import config as cnf

def read_filepaths(filepath: str=cnf.UCR_FP)-> List[Tuple[str,str]]:
    
    if get_run_mode() == "train":
        suffix = cnf.UCR_TRAIN_NAME
    else:
        suffix = cnf.UCR_TEST_NAME
    ts_infos = []
    for root, dirs, files in os.walk(filepath):
        for name in files:
            if (name.endswith(suffix)):
                path_tmp = os.path.join(root,name)
                ts_name = re.split("/", root)[-1]
                ts_infos.append((ts_name,path_tmp))

    return ts_infos

def read_ucr_csv(ts_info: Tuple[str, str]) -> pd.DataFrame:
    ts_name = ts_info[0]
    fp = ts_info[1]

    df = pd.read_csv(fp, sep="\t", header=None)
    df['name'] = ts_name
    df['no'] = df.index.values
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df = df[cols]
    return df

def set_global_df(df: pd.DataFrame):
    global df_g
    df_g = df


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

def get_stats_mp(data_t: Tuple[str,int,int,np.ndarray]) -> pd.DataFrame:
    ts_name = data_t[0]
    no = data_t[1]
    type_cls = data_t[2]
    ar = data_t[3]

    count = ar.shape[0]
    mean = np.mean(ar)
    std = np.std(ar)
    min_val = np.min(ar)
    q25 = np.quantile(ar, .25)
    q50 = np.median(ar)
    q75 = np.quantile(ar, .75)
    max_val = np.max(ar)

    period = 12
    trend_ar = get_trend(ar, period)
    m, b = fit_trend(trend_ar)

    idx = ['ts_name', 'no', 'class', 'm', 'b', 'count', 'mean', 'std', 'min',
           'q25', 'q50', 'q75', 'max']
    res = pd.Series([ts_name, no, type_cls, count, m, b, mean, std, min_val, q25, q50,\
                    q75, max_val], index=idx)
    return res


def convert_row(s: pd.Series)-> Tuple[str, int, int, np.ndarray]:
    """
    separate row into components for fourier transform and
    classification
    """
    name = s['name']
    ts_no = s['no']
    type_cls = int(s.iloc[2])
    ar = np.array(s.iloc[3:].dropna())
    return (name, ts_no, type_cls, ar)

def separate_ucr_ts(df: pd.DataFrame) -> List[Tuple[str, np.ndarray]]:
    ts_l = df.apply(convert_row, axis=1).tolist()
    return ts_l

def full_fft(ts_name: str,
             no: int,
             type_cls: int,
             ar: np.ndarray) -> pd.DataFrame:
    # full FFT
    n = ar.shape[0]
    transform_name = 'fft'
    fhat = np.abs(np.fft.fft(ar))   # compute FFT
    PSD = fhat * np.conj(fhat) / n  # power spectrum
    fft_freqs = cnf.REF_LEN/fhat.shape[0] * np.arange(n) # create x-axis for frequencies
    fft_freq_appx_idx = np.digitize(fft_freqs,freq_ranges)

    # create FFT dataframe
    df = create_df_apx(ts_name,
                       fft_freqs,
                       transform_name,
                       fhat,
                       PSD,
                       fft_freq_appx_idx)
    df['no'] = no
    df['class'] = type_cls

    return df

def hamming_fft(ts_name: str,
                no: int,
                type_cls: int,
                ar: np.ndarray) -> pd.DataFrame:
    # FFT with Hamming Window
    n = ar.shape[0]
    transform_name = 'Hamming'
    ar_hamming = ar * np.hamming(n)
    fhat_hamming = np.abs(np.fft.fft(ar_hamming))
    PSD_hamming = fhat_hamming * np.conj(fhat_hamming)
    hamming_freqs = cnf.REF_LEN/fhat_hamming.shape[0] * np.arange(n)
    hamming_freq_appx_idx = np.digitize(hamming_freqs,freq_ranges)

    # create hamming window df

    df = create_df_apx(ts_name,
                       hamming_freqs,
                       transform_name,
                       fhat_hamming,
                       PSD_hamming,
                       hamming_freq_appx_idx)
    df['no'] = no
    df['class'] = type_cls

    return df

def welch_fft(ts_name: str,
              no: int,
              type_cls: int,
              ar: np.ndarray):
    # FFT with Welch method

    n = ar.shape[0]
    transform_name = 'Welch'
    seg_length = np.floor(1/20*n)
    if seg_length == 0:
        # set minimum window for periods to short for 5% threshold
        seg_length = 10
    welch_freqs, PSD_welch = signal.welch(ar, nperseg=seg_length,
                                                     window='hamming')
    welch_freqs = cnf.REF_LEN/n * welch_freqs # adjusting to reference length of ts
    welch_freq_apx_idx = np.digitize(welch_freqs, freq_ranges)

    # create Welch window dataframe
    df = create_df_apx(ts_name,
                       welch_freqs,
                       transform_name,
                       fhat=np.array([np.nan]*welch_freqs.shape[0]),
                       PSD=PSD_welch,
                       freq_apx_idx=welch_freq_apx_idx,
                       )
    df['no'] = no
    df['class'] = type_cls
    
    return df
                       
    

def create_df_apx(ts_name: str,
                  freqs: np.ndarray,
                  transform_name: str,
                  fhat: np.ndarray,
                  PSD: np.ndarray,
                  freq_apx_idx: np.ndarray
                  ) -> pd.DataFrame:
    
    df = pd.DataFrame({
        'ts_name': [ts_name]*freqs.shape[0],
        'type': [transform_name]*freqs.shape[0],
        'fhat': fhat.real,
        'PSD': PSD.real,
        'freq': freqs,
        'freq_apx_idx': freq_apx_idx,
        'freq_apx': freq_ranges[freq_apx_idx]
    })
    return df

def compute_fr_ranges(ts_t: Tuple[str, int,
                                  int, np.array]) -> pd.DataFrame:
    ts_name = ts_t[0]
    no = ts_t[1]
    type_cls = ts_t[2]
    ar = ts_t[3]

    
    # full FFT
    df_fft = full_fft(ts_name, no, type_cls, ar)
    # Hamming window (global)
    df_hamming = hamming_fft(ts_name, no, type_cls, ar)
    # Welch's method (hamming window)
    df_welch = welch_fft(ts_name=ts_name,
                         no=no,
                         type_cls=type_cls,
                         ar=ar)

    # combine results
    df_res = pd.concat([df_fft, df_hamming, df_welch])
    
    # zero frequencies need to stay 0
    df_res.loc[df_res['freq']==0, 'freq_apx'] = 0

    return df_res

def set_franges(start: int,
                stop: int,
                steps: int):
    global freq_ranges
    freq_ranges = 10**np.linspace(start,
                                  stop,
                                  steps)

def transform_raw_ts(start_time: datetime.date,
                     no_prc: int,
                     df_l: List[pd.DataFrame])-> pd.DataFrame:
    """
    transform raw series into fourier space and retain name and classification
    """
    tnf_csv_fp = cnf.UCR_FFT_APX_FP + "_" + get_run_mode() + ".csv"
    tnf_csv_path = Path(tnf_csv_fp)

    if tnf_csv_path.is_file():
        print("reading file from storage")
        df_approx = pd.read_csv(tnf_csv_fp)
        file_read_time = datetime.now()-start_time
        print("file read in {}".format(file_read_time))
        return df_approx
    else:

        print("Starting Fourier transform")
        approx_l = []
        start = -1
        stop = 3
        steps = 410
        with Pool(processes=no_prc,
                  initializer=set_franges,
                  initargs=(start, stop, steps)) as pool:
            for res in tqdm(pool.imap_unordered(compute_fr_ranges, df_l,
                                                chunksize=50),
                            total=len(df_l)):

                approx_l.append(res)

        print("FFT computation finished after: {}".format(datetime.now()-start_time))

        df_approx = pd.concat(approx_l)
        # clean up column order
        cols = df_approx.columns.to_list()
        cols = cols[:1]+cols[-2:]+cols[1:-2]

        # write to file:
        df_approx.to_csv(tnf_csv_fp, index=False)
        print("Frequencies approximation file written after: {}".format(datetime.now()-start_time))
        
        return df_approx

def set_run_mode(rm: str):
    global run_mode
    run_mode = rm

def get_run_mode() -> str:
    return run_mode

def reduce_to_top_frequencies(start_time: datetime.date,
                              df: pd.DataFrame,
                              n:int = cnf.NO_TOP_FREQ) -> pd.DataFrame:
    """
    reduce frequencies to n most powerful frequencies
    """
    function_time = datetime.now()
    N = cnf.NO_TOP_FREQ

    freq_l_fp = cnf.UCR_FREQ_L_FP + "_" + get_run_mode() + ".csv"
    freq_l_path = Path(freq_l_fp)
    
    if freq_l_path.is_file():
        print("reading csv with top frequencies for all time series")
        df_res = pd.read_csv(freq_l_fp)

        csv_read = datetime.now()-function_time
        print("Frequencies read after {}".format(csv_read))
        print("Total time elapsed: {}".format(datetime.now()-start_time))
        
        return df_res
    else:
        print("starting reduction to top {} frequencies".format(N))

        ts_l = []
        df_res = pd.DataFrame(columns=['ts_name', 'freq_ids'])
        for ts in tqdm(df.groupby(["type", "ts_name", "no", "class"])):
            ts_type = ts[0][0]
            ts_name = ts[0][1]
            no = ts[0][2]
            cls_type = ts[0][3]
            df_sub = ts[1]

            # ensure freq ids are kept as integer
            df_sub['freq_apx_idx'] = df_sub['freq_apx_idx'].astype(int)

            freq_l = df_sub['freq_apx_idx'].tolist()
            PSD_l = df_sub['PSD'].tolist()

            # returns largest index positions sorted from smallest to biggest
            try:
                idx_powerful_PSD = sorted(range(len(PSD_l)), key= lambda
                                      x: PSD_l[x])[-N:]
            except TypeError:
                print("ts_type: {}\nts_name: {}\nno: {}\ncls_type: {}"\
                      .format(ts_type,ts_name,no,cls_type))


            # get the largest values by their index into the list
            freq_idx = [freq_l[i] for i in idx_powerful_PSD]

            df_res = df_res.append(pd.DataFrame({'ts_name': ts_name,
                                                 'type': ts_type,
                                                 'no': no,
                                                 'class': cls_type,
                                                 'freq_ids': str(freq_idx)},
                                                index=[0]))


        df_res.reset_index(inplace=False)
        df_res.to_csv(freq_l_fp, index=False)
        print("frequencies written to file in {}".format(datetime.now()-function_time))
        print("Total elapsed time: {}".format(datetime.now()-start_time))

        return df_res

def compute_ts_stats(start_time: datetime.date,
                     no_prc: int,
                     df_top: pd.DataFrame,
                     time_series: List[pd.DataFrame])-> pd.DataFrame:
    """
    fit the trend of time series decomposition and simple time series statistics
    """
    function_time = datetime.now()

    ts_stats_fp = cnf.UCR_STATS_FP + "_" + get_run_mode() + ".csv"
    ts_stats_path = Path(ts_stats_fp)

    if ts_stats_path.is_file():
        print("reading time series stats from file")
        df = pd.read_csv(ts_stats_fp)
        
        csv_read = datetime.now()-function_time
        print("time series stats read in {}".format(csv_read))

        return df
    else:
        # convert time series - and create global df
        series_l = [pd.Series(tup) for tup in time_series ]
        df_all = pd.DataFrame(series_l)
        tqdm.pandas()
        # read reference data
        print("starting multiprocessing for statistical KPI")
        res_l = []
        with Pool(processes=no_prc) as pool:
            for res in tqdm(pool.imap_unordered(get_stats_mp, time_series,
                                                chunksize=10),
                            total=len(series_l)):
                res_l.append(res)

        print("Multiprocessing completed, runtime: {}".format(datetime.now()-function_time))
        df_stats = pd.DataFrame(res_l)
        merge_l = ['ts_name', 'no', 'class']
        df_res = df_top.merge(df_stats,
                              left_on=merge_l,
                              right_on=merge_l,
                              how='left')
        
        print("Result dataframe created, runtime: {}".format(datetime.now()-function_time))


        df_res.to_csv(ts_stats_fp, index=False)

        stats_created_time = datetime.now()-function_time
        print("statistics and trend fit created in: {}".format(stats_created_time))

        return df_res


def read_raw_ts(start_time: datetime.date,
                no_prc: int)->pd.DataFrame:
    """
    reading raw time series
    """
    ts_infos = read_filepaths()
    # read csv files
    print("starting to read raw csv")
    df_l = []
    with Pool(processes=no_prc) as pool:
        for res in tqdm(pool.imap_unordered(read_ucr_csv, ts_infos),
                        total=len(ts_infos)):
            df_l.append(res)

    csvs_read = datetime.now()-start_time
    print("CSVs read after: {}".format(csvs_read))

    time_series = []
    with Pool(processes=no_prc) as pool:
        for res in tqdm(pool.imap_unordered(separate_ucr_ts, df_l,
                                            chunksize=14),
                        total=len(df_l)):
            time_series += res

    split_complete = datetime.now()-csvs_read
    print("Split of dataframes into list of time series completed in: {}".format(split_complete))

    return time_series

def run_test_mode() -> bool:
    """
    retrieve program arguments
    """
    parser = argparse.ArgumentParser(description="Determine whether to run on train or test test")
    parser.add_argument("--test", dest="run_test", action='store_true', default=False,
                        help="execute on test data set")

    args = parser.parse_args()

    return args.run_test

def main()->None:
    if run_test_mode():
        # determine params for test mode
        set_run_mode("test")
    else:
        # determine params for train mode
        set_run_mode("train")

    start_time = datetime.now()
    no_prc = cpu_count()-1

    time_series = read_raw_ts(start_time, no_prc)
    
    df_apx = transform_raw_ts(start_time,
                              no_prc,
                              time_series)

    df_freq_l = reduce_to_top_frequencies(start_time,
                                          df_apx)

    df_stats = compute_ts_stats(start_time, no_prc,
                                df_freq_l, time_series)

    print("computation completed after: {}".format(datetime.now()-start_time))


if __name__ == "__main__":
    main()
