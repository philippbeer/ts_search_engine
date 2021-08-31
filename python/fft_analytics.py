from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm

def read_csv(data: str)-> pd.DataFrame:
    """
    reads csv to df
    """
    return pd.read_csv(data)


def split_df(df: pd.DataFrame,
             no_items: int = 100) -> List[pd.DataFrame]:
    l = df.shape[0]
    start = 0
    stop = increment = int(np.floor(l/no_items))
    
    df_l = []
    while (stop<df.shape[0]):
        df_tmp = df[start:stop]
        df_l.append(df_tmp)
        start = stop
        stop += increment

    # add missing last rows
    df_tmp = df[start:]
    df_l.append(df_tmp)

    return df_l


def separate_ts(df: pd.DataFrame) -> List[Tuple[str, np.ndarray]]:
    ts_l = []
    for ts_name in df.iloc[:,0]:
        ar = np.array(df[df['V1']==ts_name].iloc[:,1:].dropna(axis=1))[0]
        ts_l.append((ts_name, ar))
    return ts_l
    
    
# plotting of power spectrum for each
def plot_psd(data: Tuple[str,pd.DataFrame]) -> str:
    fig, axes = plt.subplots(1, 1, sharey=True)
    axes.set_xscale('log')
    for ts_name in tqdm(data[1].iloc[:,0]):
        ar = np.array(data[1][data[1]['V1']==ts_name].iloc[:,1:].dropna(axis=1))[0]
        n = ar.shape[0]
        dt = 0.001
        fhat = np.abs(np.fft.fft(ar))  # compute the FFT
        PSD = fhat * np.conj(fhat) / n # power spectrum (power per frequency)
        freq = (1/(dt*n)) * np.arange(n) # create x-axis of frequencies
        L = np.arange(1,np.floor(n/2), dtype='int')
    
        # adding plot
        plt.plot(freq[L], PSD[L], "o-", linewidth=2, label=ts_name)
    title = data[0] + "Power Spectrum"
    fig.suptitle(title)
    plt.savefig(data[0]+"_psd.png")
    return data[0]

def full_fft(ts_name:str,
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
    return df

def hamming_fft(ts_name: str,
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

    return df

def welch_fft(ts_name: str,
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
        'fhat': fhat,
        'PSD': PSD,
        'freq': freqs,
        'freq_apx_idx': freq_apx_idx,
        'freq_apx': freq_ranges[freq_apx_idx]
    })
    return df

def compute_fr_ranges(ts_t: Tuple[str,np.array]) -> pd.DataFrame:
    dt = 0.001
    ts_name = ts_t[0]
    ar = ts_t[1]

    
    # full FFT
    df_fft = full_fft(ts_name, ar, dt)
    # Hamming window (global)
    df_hamming = hamming_fft(ts_name, ar, dt)
    # Welch's method (hamming window)
    df_welch = welch_fft(ts_name, ar, dt)

    # combine results
    df_res = pd.concat([df_fft, df_hamming, df_welch])
    
    # zero frequencies need to stay 0
    df_res.loc[df_res['freq']==0, 'freq_apx'] = 0

    return df_res

def compress_series(fhat: np.array,
                    PSD: np.array) -> np.array:
    thresh = np.percentile(PSD, .9)
    idx = PSD > thresh
    PSDcore = PSD * idx
    fhat = idx * fhat
    ffilt = np.fft.ifft(fhat)
    return (PSDcore, ffilt)

def set_franges(start: int,
                stop: int,
                steps: int):
    global freq_ranges
    freq_ranges = 10**np.linspace(start,
                                  stop,
                                  steps)
    

def main():
    start_time = datetime.now()
    # loading m4 competition data

    hourly_fp = "../m4_data/hourly-train.csv"
    # daily_fp = "../m4_data/Daily-train.csv"
    # weekly_fp = "../m4_data/Weekly-train.csv"
    # monthly_fp = "../m4_data/Monthly-train.csv"
    # quarterly_fp = "../m4_data/Quarterly-train.csv"
    yearly_fp = "../m4_data/Yearly-train.csv"
                        
    # m4_l = [hourly_fp, daily_fp, weekly_fp,
    #     monthly_fp, quarterly_fp, yearly_fp]
    m4_l = [hourly_fp, yearly_fp]

    no_prc = cpu_count()-1

    # read csv files
    print("starting to read csv")
    df_l = []
    with Pool(processes=no_prc) as pool:
        for res in tqdm(pool.imap_unordered(read_csv, m4_l),
                        total=len(m4_l)):
            df_l.append(res)

    print("CSVs read after: {}".format(datetime.now()-start_time))

    print("concatenating dfs")
    df = pd.concat(df_l)
    df = df.sample(10000)
    df_l = None

    df_l = split_df(df)

    print("concatenation completed after: {}".format(datetime.now()-start_time))

    # creating time series tuple list
    print("splitting df into list of ts")
    ts_l = []
    with Pool(processes=no_prc) as pool:
        for res in tqdm(pool.imap_unordered(separate_ts, df_l,
                                            chunksize=14),
                        total=len(df_l)):
            ts_l += res

    print("dfs split after: {}".format(datetime.now()-start_time))

    # starting pool
    print("starting multiprocessing for FFT")
    approx_l = []
    start = -1
    stop = 3
    steps = 410
    with Pool(processes=no_prc,
              initializer=set_franges,
              initargs=(start, stop, steps)) as pool:
        for res in tqdm(pool.imap_unordered(compute_fr_ranges, ts_l,
                                            chunksize=50),
                        total=len(ts_l)):

            approx_l.append(res)

    print("FFT computation finished after: {}".format(datetime.now()-start_time))

    df_approx = pd.concat(approx_l)

    df_approx.to_csv("../data/df_apx_win.csv", index=False)

    print("computation completed after: {}".format(datetime.now()-start_time))

if __name__ == "__main__":
    main()
    
