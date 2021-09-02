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
