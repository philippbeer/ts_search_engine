from math import ceil
from random import choices

from dtw import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

def read_ts_data() -> pd.DataFrame:
    df_hourly = pd.read_csv("../m4_data/Hourly-train.csv")
    print("hourly: {}".format(df_hourly.shape))
    df_daily = pd.read_csv("../m4_data/Daily-train.csv")
    print("daily: {}".format(df_daily.shape))
    df_weekly = pd.read_csv("../m4_data/Weekly-train.csv")
    print("weekly: {}".format(df_weekly.shape))
    df_monthly = pd.read_csv("../m4_data/Monthly-train.csv")
    print("monthly: {}".format(df_monthly.shape))
    df_quarterly = pd.read_csv("../m4_data/Quarterly-train.csv")
    print("quarterly: {}".format(df_quarterly.shape))
    df_yearly = pd.read_csv("../m4_data/Yearly-train.csv")
    print("yearly: {}".format(df_yearly.shape))

    df_all = pd.concat([df_hourly,
                   df_daily,
                   df_weekly,
                    df_monthly,
                   df_quarterly,
                   df_yearly])

    df_hourly = None
    df_daily = None
    df_weekly = None
    df_monthly = None
    df_quarterly = None
    df_yearly = None

    df_all.reset_index(drop=True, inplace=True)

    return df_all

def write_plots(df_winners: pd.DataFrame,
                df_all: pd.DataFrame) -> None:
    for elm in tqdm(df_winners.groupby("ts_name")):
        src_name = elm[0]
        df_sub = elm[1]
    
        fft_name = df_sub[df_sub['type']=='fft']['candidate'].values[0]
        ham_name = df_sub[df_sub['type']=='Hamming']['candidate'].values[0]
        wel_name = df_sub[df_sub['type']=='Welch']['candidate'].values[0]
    
        ar_src = np.array(df_all[df_all['V1']==src_name].iloc[:,1:].dropna(axis=1))[0]
        ar_fft = np.array(df_all[df_all['V1']==fft_name].iloc[:,1:].dropna(axis=1))[0]
        ar_ham = np.array(df_all[df_all['V1']==ham_name].iloc[:,1:].dropna(axis=1))[0]
        ar_wel = np.array(df_all[df_all['V1']==wel_name].iloc[:,1:].dropna(axis=1))[0]
    
        plt.plot(ar_src, '-', label="Src: " + src_name)
        plt.plot(ar_fft, label="FFT: " + fft_name)
        plt.plot(ar_ham, label="Hamming: "+ ham_name)
        plt.plot(ar_wel, label="Welch: " + wel_name)
        plt.legend()
        plt.title("Match per Window Type for: " +src_name)
        plt.savefig("../img/wd_1000_winners/" + src_name + ".png")
        plt.close()

def main():
    df_all = read_ts_data()

    df_winners = pd.read_csv("../data/df_winners_1000.csv")
    write_plots(df_winners, df_all)
    

if __name__ == "__main__":
    main()
