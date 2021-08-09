from multiprocessing import Pool, cpu_count
import os
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# loading m4 competition data

hourly_fp = "m4_data/hourly-train.csv"
daily_fp = "m4_data/Daily-train.csv"
weekly_fp = "m4_data/Weekly-train.csv"
monthly_fp = "m4_data/Monthly-train.csv"
quarterly_fp = "m4_data/Quarterly-train.csv"
yearly_fp = "m4_data/Yearly-train.csv"
                        
# m4_names = ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]


m4_data = {"hourly": hourly_fp,
           "daily": daily_fp,
           "weekly": weekly_fp,
           "monthly": monthly_fp,
           "quarterly": quarterly_fp,
           "yearly": yearly_fp}

def read_csv(data: Tuple[str,str])-> Tuple[str,pd.DataFrame]:
    """
    reads csv to df
    """
    return (data[0], pd.read_csv(data[1]))

    

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

def compute_fr_ranges(ts_t: Tuple[str,np.array,np.array]) -> pd.DataFrame:
    dt = 0.001
    ts_name = ts_t[0]
    ar = ts_t[1]
    f_ranges = ts_t[2]
    n = ar.shape[0]
    fhat = np.abs(np.fft.fft(ar))   # compute FFT
    PSD = fhat * np.conj(fhat) / n  # power spectrum
    freq = (1/(dt*n)) * np.arange(n) # create x-axis for frequencies
    freq_approx_idx = np.digitize(freq,f_ranges)
    L = np.arange(1,np.floor(n/2), dtype='int')
    # create df_approx for comparison
    df_res = pd.DataFrame({'ts_name': [ts_name]*len(freq),
                              'fhat': fhat,
                              'PSD': PSD,
                              'freq': freq,
                           'freq_approx_idx': freq_approx_idx,
                         'freq_approx': f_ranges[freq_approx_idx]})
    # zero frequencies need to stay 0
    df_res.loc[df_res['freq']==0, 'freq_approx'] = 0
    return df_res

def concat_dfs(df_l: List[Tuple[str,str]]) -> pd.DataFrame:
    frames = [elm[1] for elm in df_l]
    return pd.concat(frames)

def compress_series(fhat: np.array,
                    PSD: np.array) -> np.array:
    thresh = np.percentile(PSD, .9)
    idx = PSD > thresh
    PSDcore = PSD * idx
    fhat = idx * fhat
    ffilt = np.fft.ifft(fhat)
    return (PSDcore, ffilt)

def compare_approximation():
    pass
    

def main():
    no_prc = cpu_count()-1

    file_list = [(k,v) for k,v in m4_data.items()]
    with Pool(processes=no_prc) as pool:
        df_list = pool.map(read_csv, file_list)

    # with Pool(processes=no_prc) as pool:
    #     pool.map(plot_psd, df_list)
    df = concat_dfs(df_list)

    ts_l = []
    f_ranges = 10**np.linspace(-1,3,410)
    for ts_name in tqdm(df.iloc[:,0]):
        ar = np.array(df[df['V1']==ts_name].iloc[:,1:].dropna(axis=1))[0]
        ts_l.append((ts_name, ar, f_ranges))
    approx_l = []
    with Pool(processes=no_prc) as pool:
        for res in tqdm(pool.imap_unordered(compute_fr_ranges, ts_l)):
            approx_l.append(res)
        # approx_l = pool.map(compute_fr_ranges, ts_l)

    df_approx = pd.concat(approx_l)

    df_approx.to_csv("df_approx.csv", index=False)


if __name__ == "__main__":
    main()
    
