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

def read_ucr():
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
    df_ucr_test = pd.DataFrame()

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

    for ts_info in tqdm(ts_test_infos):
        ts_name = ts_info[0]
        fp = ts_info[1]

        df_tmp = pd.read_csv(fp, sep='\t', header=None)
        df_tmp['name'] = ts_name
        df_tmp['no'] = df_tmp.index
        cols = df_tmp.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df_tmp = df_tmp[cols]
        df_ucr_test = df_ucr_test.append(df_tmp)


    print("df_train shape: {}".format(df_ucr_train.shape))
    print("df_test shape: {}".format(df_ucr_test.shape))
    return df_ucr_train, df_ucr_test

def read_dtw_results():
    return pd.read_csv("../data/df_dtw_comp.csv")

def get_missing_categories(df_test, df_dtw):
    cats_all = set(df_test['name'].unique())
    cats_matched = set(df_dtw[df_dtw['ts_1']==df_dtw['ts_2']]['ts_1'].unique())
    cats = list(cats_all-cats_matched)
    return cats

def create_vis(cats, df_train, df_test):
    fig, axs = plt.subplots(16,2,figsize=(20,20))
    fig.tight_layout()

    for i in tqdm(range(len(cats))):
        cat = cats[i]
        df_train_tmp = df_train[df_train['name']==cat]
        df_test_tmp = df_test[df_test['name']==cat]

        for key, row in df_train_tmp.iterrows():
            name = row.iloc[0]+" - "+str(row.iloc[1])+" - "+str(row.iloc[2])
            ar = np.array(row.iloc[3:].dropna())
            n = ar.size
            x = np.linspace(1,n,n)
            sns.lineplot(x=x, y=ar, label=name, ax=axs.flatten()[i])

        for key, row in df_test_tmp.iterrows():
            name = row.iloc[0]+" - "+str(row.iloc[1])+" - "+str(row.iloc[2])
            ar = np.array(row.iloc[3:].dropna())
            n = ar.size
            x = np.linspace(1,n,n)
            sns.lineplot(x=x, y=ar, label=name, ax=axs.flatten()[i+1])  
    fig.savefig("../img/dtw_missing_categories.png")
    
def main():
    start = time.time()
    df_train, df_test = read_ucr()
    print(f"UCR raw read in {time.time()-start:2}")
    df_dtw = read_dtw_results()
    print(f"DTW results read after: {time.time()-start}")
    cats= get_missing_categories(df_test, df_dtw)
    print(f"Missing categories found after: {time.time()-start}")
    print("starting vis")
    create_vis(cats, df_train,df_test)
    print(f"vis completed after: {time.time()-start}")

if __name__ == "__main__":
    main()
