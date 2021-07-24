# set working directory
setwd("~/workspace/UNIC/comp-593/r_ts_analysis/")
library(readr)
df <- read_csv("../m4_data/Hourly-train.csv")

row <- df[169,2:ncol(df)]
row <- as.vector(remove_empty(row))

row2 <- df[170,2:ncol(df)]
row2 <- as.vector(remove_empty(row2))