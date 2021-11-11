import math
import os
import re
import time
from typing import List

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from tqdm import tqdm


from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def read_m4()->pd.DataFrame:
    df_hourly = pd.read_csv("../m4_data/Hourly-train.csv")
    print("hourly read: {}".format(df_hourly.shape))
    df_daily = pd.read_csv("../m4_data/Daily-train.csv")
    print("daily read: {}".format(df_daily.shape))
    df_weekly = pd.read_csv("../m4_data/Weekly-train.csv")
    print("weekly: {}".format(df_weekly.shape))
    df_monthly = pd.read_csv("../m4_data/Monthly-train.csv")
    print("monthly: {}".format(df_monthly.shape))
    df_quarterly = pd.read_csv("../m4_data/Quarterly-train.csv")
    print("quarterly: {}".format(df_quarterly.shape))
    df_yearly = pd.read_csv("../m4_data/Yearly-train.csv")
    print("yearly: {}".format(df_yearly.shape))

    df_m4_raw = pd.concat([df_hourly,
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

    return df_m4_raw

def read_ucr()->pd.Series:
    ts_train_infos = []
    ts_test_infos = []
    for root, dirs, files in os.walk("../data/ucr_data/UCRArchive_2018/"):
        for name in files:
            if(name.endswith("_TRAIN.tsv")):
                path_tmp = os.path.join(root,name)
                ts_name = re.split("/", root)[-1]
                ts_train_infos.append((ts_name, os.path.join(root,name)))
            elif(name.endswith("_TEST.tsv")):
                path_tmp = os.path.join(root,name)
                ts_name = re.split("/", root)[-1]
                ts_test_infos.append((ts_name, os.path.join(root,name)))
    df_ucr_train = pd.DataFrame()


    for ts_info in tqdm(ts_train_infos):
        ts_name = ts_info[0]
        fp = ts_info[1]
    
        df_tmp = pd.read_csv(fp, sep='\t', header=None)
        df_tmp['name'] = ts_name
        df_tmp['no'] = df_tmp.index
        cols = df_tmp.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df_tmp = df_tmp[cols]
        df_ucr_train = df_ucr_train.append(df_tmp)

    return df_ucr_train

def get_m4_ar(df: pd.DataFrame)->np.ndarray:
    ar_m4 = df.iloc[:,3:].to_numpy()
    ar_m4 = ar_m4.ravel()
    print(ar_m4.shape)
    ar_m4 = ar_m4[~np.isnan(ar_m4)]
    print(ar_m4.shape)
    return ar_m4

def get_ucr_ar(df: pd.DataFrame)->np.ndarray:
    ar_ucr = df.iloc[:,3:].to_numpy()
    ar_ucr = ar_ucr.ravel()
    print(ar_ucr.shape)
    ar_ucr = ar_ucr[~np.isnan(ar_ucr)]
    print(ar_ucr.shape)
    return ar_ucr

def main():
    start = time.time()
    df_m4 = read_m4()
    print("M4 read: {}".format(time.time()-start))

    df_ucr = read_ucr()
    print("UCR read: {}".format(time.time()-start))
    ar_m4 = get_m4_ar(df_m4)
    print("M4 array created: {}".format(time.time()-start))
    ar_ucr = get_ucr_ar(df_ucr)
    print("UCR array created: {}".format(time.time()-start))

    sns.set(font_scale=1.5)
    fig, axs = plt.subplots(2,1, figsize=(13,9))
    fig.suptitle("Value distribution - M4 / UCR repositories")
    print("starting histogram")
    sns.histplot(x=ar_m4, ax=axs.flatten()[0], bins='fd')
    print("hist 1 done: {}".format(time.time()-start))
    print("starting hist 2")
    sns.histplot(x=ar_ucr, ax=axs.flatten()[1], bins='fd')
    axs.flatten()[0].set_yscale('log')
    axs.flatten()[0].set_xlabel("M4 value range")
    axs.flatten()[1].set_yscale('log')
    axs.flatten()[1].set_xlabel("UCR value range")
    fig.savefig("../img/m4_ucr_val_hist.png")
    print("done in {}".format(time.time()-start))

    


if __name__ == "__main__":
    main()
