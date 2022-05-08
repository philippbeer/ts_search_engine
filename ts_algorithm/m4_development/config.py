from typing import Literal

# data file paths
HOURLY_FP = "../../m4_data/hourly-train.csv"
DAILY_FP = "../../m4_data/Daily-train.csv"
WEEKLY_FP = "../../m4_data/Weekly-train.csv"
MONTHLY_FP = "../../m4_data/Monthly-train.csv"
QUARTERLY_FP = "../../m4_data/Quarterly-train.csv"
YEARLY_FP = "../../m4_data/Yearly-train.csv"

M4_FP_L = [HOURLY_FP,
           DAILY_FP,
           WEEKLY_FP,
           MONTHLY_FP,
           QUARTERLY_FP,
           YEARLY_FP]
M4_FP_L_TEST = [HOURLY_FP, WEEKLY_FP]

TS_STATS_FP = "../data/df_stats.csv"

# UCR Folder
UCR_FP = "../data/ucr_data/UCRArchive_2018/"

UCR_TRAIN_NAME = "_TRAIN.tsv"
UCR_TEST_NAME = "_TEST.tsv"

UCR_FFT_APX_FP = "../data/df_ucr_apx_win"
UCR_FREQ_L_FP = "../data/df_ucr_freq_l"
UCR_STATS_FP = "../data/df_ucr_stats"

UCR_TS_STATS_TRAIN_FP = "../data/df_ucr_stats_train.csv"
UCR_TS_STATS_TEST_FP = "../data/df_ucr_stats_test.csv"

UCR_MATCH_RES_FP = "../data/df_ucr_match_scores_samples.csv"
UCR_MIN_DELTA_FP = "../data/df_ucr_min_delta.csv"
UCR_DELTA_KPI = ['d_m', 'd_mean', 'd_std', 'd_q25', 'd_q50', 'd_q75', 'd_min', 'd_max']

# Period Mapping
PERIOD_MAPPING = {
    'H': 24,
    'D': 7,
    'W': 52,
    'M': 12,
    'Q': 4,
    'Y': 1
}

# Window types
FFT = 'fft'
HAM = 'Hamming'
WELCH = 'Welch'

# Threshold
MEAN_THRESH =2.5

# FFT config
NO_TOP_FREQ = 5

# Matching criteria frequencies
MATCH_SCORE_THRESH = 10**3

# Ref Frequency Length
REF_LEN = 176
