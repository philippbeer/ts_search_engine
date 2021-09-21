import argparse
from functools import partial
import time
import math
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import re
from typing import List, Tuple, Union

from dtw import dtw;
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
    """
    reading filepaths to dataframe and filter if applicable
    """
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
    df_g = df.transpose()


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

def separate_ucr_ts(df: pd.DataFrame) -> List[Tuple[str, int, int, np.ndarray]]:
    ts_l = df.apply(convert_row, axis=1).tolist()
    return ts_l

def full_fft(ts_name: str,
             no: int,
             type_cls: int,
             ar: np.ndarray,
             dt: float) -> pd.DataFrame:
    # full FFT
    n = ar.shape[0]
    transform_name = 'fft'
    fhat = np.abs(np.fft.fft(ar))   # compute FFT
    PSD = fhat * np.conj(fhat) / n  # power spectrum
    fft_freqs = (1/(dt*n)) * np.arange(n) # create x-axis for frequencies
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
                ar: np.ndarray,
                dt: float) -> pd.DataFrame:
    # FFT with Hamming Window
    n = ar.shape[0]
    transform_name = 'Hamming'
    ar_hamming = ar * np.hamming(n)
    fhat_hamming = np.abs(np.fft.fft(ar_hamming))
    PSD_hamming = fhat_hamming * np.conj(fhat_hamming)
    hamming_freqs = (1/(dt*n)) * np.arange(n)
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
              ar: np.ndarray,
              dt: float):
    # FFT with Welch method

    n = ar.shape[0]
    transform_name = 'Welch'
    seg_length = np.floor(1/20*n)
    if seg_length == 0:
        # set minimum window for periods to short for 5% threshold
        seg_length = 10
    welch_freqs, PSD_welch = signal.welch(ar, nperseg=seg_length,
                                                     window='hamming')
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
    dt = 0.001
    ts_name = ts_t[0]
    no = ts_t[1]
    type_cls = ts_t[2]
    ar = ts_t[3]

    
    # full FFT
    df_fft = full_fft(ts_name, no, type_cls, ar, dt)
    # Hamming window (global)
    df_hamming = hamming_fft(ts_name, no, type_cls, ar, dt)
    # Welch's method (hamming window)
    df_welch = welch_fft(ts_name=ts_name,
                         no=no,
                         type_cls=type_cls,
                         ar=ar,
                         dt=dt)

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

def dtw_match(series: Tuple[str, int, int, np.ndarray])->pd.Series:
    """
    find the closest time series time series via dtw
    """
    # df_sub = df_g
    ts_name = series[0]
    ts_no = series[1]
    cls_type = series[2]
    ts = series[3]

    print("running dtw match for: {} - {}".format(ts_name,ts_no), flush=True)
    
    # tqdm.pandas()
    sub_series = df_g.apply(apply_dtw, args=(ts,)).reset_index(drop=True)
    idx_min = sub_series.idxmin()
    series_min = df_g.iloc[:,idx_min]

    s_res = pd.Series({
        'ts_1': ts_name,
        'no_1': ts_no,
        'class_1': cls_type,
        'ts_2': series_min.loc['name'],
        'no_2': series_min.loc['no'],
        'class_2': series_min.loc[0], # this accesses column named 0 / not index 0
        'min_dtw_dist': sub_series.min()
    })
    return s_res
    

def apply_dtw(s_test: pd.Series,
              ar_tmpl: np.ndarray):
    """
    apply dtw to template and test array
    """
    ar_test = np.array(s_test.iloc[3:].dropna().to_numpy())
    aln = dtw(ar_test, ar_tmpl,
              distance_only=True)
    
    return aln.distance
    

def transform_raw_ts(start_time: float,
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
        file_read_time = time.time()-start_time
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

        print("FFT computation finished after: {:.2f}".format(time.time()-start_time))

        df_approx = pd.concat(approx_l)
        # clean up column order
        cols = df_approx.columns.to_list()
        cols = cols[:1]+cols[-2:]+cols[1:-2]

        # write to file:
        df_approx.to_csv(tnf_csv_fp, index=False)
        print("Frequencies approximation file written after: {:.2f}".format(time.time()-start_time))
        
        return df_approx

def set_run_mode(rm: str):
    global run_mode
    run_mode = rm

def get_run_mode() -> str:
    return run_mode

def reduce_to_top_frequencies(start_time: float,
                              df: pd.DataFrame,
                              n:int = cnf.NO_TOP_FREQ) -> pd.DataFrame:
    """
    reduce frequencies to n most powerful frequencies
    """
    function_time = time.time()
    N = cnf.NO_TOP_FREQ

    freq_l_fp = cnf.UCR_FREQ_L_FP + "_" + get_run_mode() + ".csv"
    freq_l_path = Path(freq_l_fp)
    
    if freq_l_path.is_file():
        print("reading csv with top frequencies for all time series")
        df_res = pd.read_csv(freq_l_fp)

        csv_read = time.time()-function_time
        print("Frequencies read after {:.2f}".format(csv_read))
        print("Total time elapsed: {:.2f}".format(time.time()-start_time))
        
        return df_res
    else:
        print("starting reduction to top {:.2f} frequencies".format(N))

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
        print("frequencies written to file in {:.2f}".format(time.time()-function_time))
        print("Total elapsed time: {:.2f}".format(time.time()-start_time))

        return df_res

def compute_ts_stats(start_time: float,
                     no_prc: int,
                     df_top: pd.DataFrame,
                     time_series: List[pd.DataFrame])-> pd.DataFrame:
    """
    fit the trend of time series decomposition and simple time series statistics
    """
    function_time = time.time()

    ts_stats_fp = cnf.UCR_STATS_FP + "_" + get_run_mode() + ".csv"
    ts_stats_path = Path(ts_stats_fp)

    if ts_stats_path.is_file():
        print("reading time series stats from file")
        df = pd.read_csv(ts_stats_fp)
        
        csv_read = time.time()-function_time
        print("time series stats read in {:.2f}".format(csv_read))

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

        print("Multiprocessing completed, runtime: {:.2f}".format(time.time()-function_time))
        df_stats = pd.DataFrame(res_l)
        merge_l = ['ts_name', 'no', 'class']
        df_res = df_top.merge(df_stats,
                              left_on=merge_l,
                              right_on=merge_l,
                              how='left')
        
        print("Result dataframe created, runtime: {:.2f}".format(time.time()-function_time))


        df_res.to_csv(ts_stats_fp, index=False)

        stats_created_time = time.time()-function_time
        print("statistics and trend fit created in: {:.2f}".format(stats_created_time))

        return df_res


def read_raw_ts(t1: float,
                no_prc: int,
                filter: Union[None,List[Tuple]] = None)\
                ->Union[pd.DataFrame,List[Tuple[str, int,\
                                                int, np.ndarray]]]:
    """
    reading raw time series
    """
    ts_infos = read_filepaths()
    # read csv files
    print("###### beginning reading raw csv ######")
    df_l = []

    with Pool(processes=no_prc) as pool:
        for res in tqdm(pool.imap_unordered(read_ucr_csv, ts_infos),
                        total=len(ts_infos)):
            df_l.append(res)

    t2 = time.time()-t1
    print("###### CSVs read after: {:.2f} ######".format(t2))

    df = pd.concat(df_l)
    
    if filter:
        df = df[df.set_index(['name', 'no']).index.isin(filter)]
        return df
    else:
        return df

def split_df(df: pd.DataFrame,
             pieces: int=7) -> List[pd.DataFrame]:
    """
    splitting dataframe into list of dataframes
    """
    stop = step = math.floor(df.shape[0]/pieces)
    start = 0

    df_l = []
    while(stop<df.shape[0]):
        df_tmp = df[start:stop]
        df_l.append(df_tmp)
        start = stop
        stop += step

    # capturing remaining elements
    df_tmp = df[start:]
    df_l.append(df_tmp)

    return df_l
    
def df_to_list(df: pd.DataFrame)-> List[Tuple[str, int,\
                                              int, np.ndarray]]:
    """
    convert dataframe to list of tuples
    """
    no_prc = cpu_count()-1
    t1 = time.time()
    print("###### splitting data into list of individual time series ######")
    df_l = split_df(df, no_prc)

    time_series = []
    with Pool(processes=no_prc) as pool:
        for res in tqdm(pool.imap_unordered(separate_ucr_ts, df_l),
                        total=len(df_l)):
            time_series += res

    print("###### Split of dataframes into list of time series completed in: {:.2f} ######".format(time.time()-t1)) 
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

def get_samples()->List[Tuple]:
    """
    reads the samples and creates list of tuples to filter by
    """
    t1 = time.time()
    print('###### Reading FFT matches to capture relevant samples ######')
    df = pd.read_csv(cnf.UCR_MATCH_RES_FP)
    df = df[['ts_1', 'no_1']]
    filter = list(zip(df['ts_1'], df['no_1']))
    print("###### data read and filter created in {:.2f}s ######".format(time.time()-t1))
    return filter
    

def main()->None:
    set_run_mode("train")
    t1 = time.time()
    no_prc = cpu_count()-1

    # reading train data
    df_train = read_raw_ts(t1, no_prc)
    # df_train = df_train.sample(1000)

    set_global_df(df_train)

    # reading and filtering test data
    t2 = time.time()
    set_run_mode("test")
    filt = get_samples()
    df_test = read_raw_ts(t2, no_prc, filt)

    # df_test = df_test.sample(10)
    ts_test_l = df_to_list(df_test)

    # run dtw comparison
    res_l = []

    t3 = time.time()
    # ch_size = math.floor(len(ts_test_l)/no_prc)
    print("###### Starting DTW comparison ######")
    with Pool(processes=no_prc,
              initializer=set_global_df,
              initargs=(df_train,)) as pool:
        for res in tqdm(pool.imap_unordered(dtw_match, ts_test_l),
                                            # chunksize=ch_size),
                        total=len(ts_test_l)):
            res_l.append(res)

    df_res = pd.DataFrame(res_l)
    print("###### DTW comparison completed in {:.2f}s ######".format(time.time()-t3))
    # print("df_res: ", df_res.head())

    df_res.to_csv("../data/df_dtw_comp.csv", index=False)
    print("computation completed after: {:.2f}".format(time.time()-t1))


if __name__ == "__main__":
    main()
