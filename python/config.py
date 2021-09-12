# data file paths
HOURLY_FP = "../m4_data/hourly-train.csv"
DAILY_FP = "../m4_data/Daily-train.csv"
WEEKLY_FP = "../m4_data/Weekly-train.csv"
MONTHLY_FP = "../m4_data/Monthly-train.csv"
QUARTERLY_FP = "../m4_data/Quarterly-train.csv"
YEARLY_FP = "../m4_data/Yearly-train.csv"

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
UCR_TEST_NANE = "_TEST.tsv"

UCR_FFT_APX_FP = "../data/df_ucr_apx_win"
UCR_FREQ_L_FP = "../data/df_ucr_freq_l"
UCR_STATS_FP = "../data/df_ucr_stats"

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
