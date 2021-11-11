import math
import os
import re
from typing import List

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from tqdm import tqdm


def main():
    df = pd.read_csv("../data/df_ucr_min_delta.csv")
    print("data read")
    df.sort_values(['ts_1','no_1','min_kpi', 'value'], inplace=True)
    df_min = df.groupby(['ts_1','no_1','min_kpi']).apply(find_min_kpi)
    vis(df_min)
    print("program finished")


def vis(df):
    fig, ax = plt.subplots(figsize=(13,9))
    sns.histplot(df, x='match_score', hue='type')
    plt.savefig("../img/match_score_hist.png")

def find_min_kpi(df: pd.DataFrame) -> str:
    """ find min kpi for combination """
    values = list(df['value'])
    idx = []
    for i in range(len(values)):
        if i == 0:
            idx.append(i)
        elif values[i] == idx[i-1]: ## conseqcutive element matches
            idx.append(i)
        else:
            break
            
    df_res = df.iloc[idx,:].sort_values('match_score', ascending=False)
    return df_res.iloc[0,:]



if __name__ == "__main__":
    main()
