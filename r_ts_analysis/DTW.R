# set working directory
setwd("~/workspace/UNIC/comp-593/r_ts_analysis/")
library(readr)
library(janitor)
library(tibble)
library(dplyr)
library(dtw)
df <- read_csv("../m4_data/Hourly-train.csv")


# loop to find closest series
no_cols <- ncol(df)
no_rows <- nrow(df)
df_dtw_cols <- c("qry", "tmpl")
df_dtw <- df_dtw_cols %>% purrr::map_dfc(setNames, object = list(integer()))
for (qry_row in 1:no_rows) {
  # empty tibble for results
  qry <- as.numeric(as.vector(remove_empty(df[qry_row,2:no_cols],
                                           which="cols")))
  
  tmp_shortest_dist <- 0
  
  for (tmpl_row in 1:no_rows) {
    # if rows are the same skip
    if (tmpl_row == qry_row) next
    tmpl <- as.numeric(as.vector(remove_empty(df[tmpl_row,2:no_cols],
                                              which="cols")))
    tmp_dtw <- dtw(qry,tmpl,keep=TRUE)
    if((tmp_shortest_dist == 0) | (tmp_dtw$distance < tmp_shortest_dist)) {
      #sprintf("updating row %d with row %d", qry_row, tmpl_row)
      print("#########")
      print(paste("updating row ", qry_row))
      print(paste("with row ", tmpl_row))
      tmp_shortest_dist <- tmp_dtw$distance
      tmp_row <- tmpl_row
    }
  }
  df_dtw <- df_dtw %>% add_row(tibble_row(qry = qry_row, tmpl = tmp_row))
}
