# set working directory
setwd("~/workspace/UNIC/comp-593/r_ts_analysis/")
library(readr)
library(janitor)
library(tibble)
library(dplyr)
library(dtw)

# reading M4 data
df <- read_csv("../m4_data/Hourly-train.csv")

# prepping variables
no_cols <- ncol(df)
no_rows <- nrow(df)
df_dtw_cols <- c("qry", "tmpl")
# storage for winner combinations
df_dtw <- df_dtw_cols %>% purrr::map_dfc(setNames, object = list(integer()))
# all results tibble

df_res <- tibble(
    .rows = NULL,
    qry = character(),
    tmpl = character(),
    qry_len = integer(),
    tmpl_len = integer(),
    dist = numeric(),
    ndist = numeric()
    )

# looping to find closest series for each series in all others
for (qry_row in 1:no_rows) {
  qry <- as.numeric(as.vector(remove_empty(df[qry_row,2:no_cols],
                                           which="cols")))
  
  tmp_shortest_dist <- 0
  
  for (tmpl_row in 1:no_rows) {
    # if rows are the same skip
    if (tmpl_row == qry_row) next
    tmpl <- as.numeric(as.vector(remove_empty(df[tmpl_row,2:no_cols],
                                              which="cols")))
    tmp_dtw <- dtw(qry,tmpl,keep=TRUE)
    
    # adding results to results table
    df_res <- df_res %>% add_row(tibble_row(
                      qry = as.character(df[qry_row,1]),
                      tmpl = as.character(df[tmpl_row,1]),
                      qry_len = tmp_dtw$N,
                      tmpl_len = tmp_dtw$M,
                      dist = tmp_dtw$distance,
                      ndist = tmp_dtw$normalizedDistance))
    
    if((tmp_shortest_dist == 0) | (tmp_dtw$distance < tmp_shortest_dist)) {
      #sprintf("updating row %d with row %d", qry_row, tmpl_row)
      print("#########")
      print(paste("updating row", qry_row))
      print(paste("with row", tmpl_row))
      tmp_shortest_dist <- tmp_dtw$distance
      tmp_row <- tmpl_row
    }
  }
  
  print("+++++++++ Closest Connection ++++++++++")
  print(paste("Row", qry_row, "is closest to", tmp_row))
  print("+++++++++++++++++++++++++++++++++++++++")
  df_dtw <- df_dtw %>% add_row(tibble_row(qry = qry_row, tmpl = tmp_row))
}

# write results to csv
write.csv(df_res, file="analytics/df_res.csv")

dtw_viz_nearest_ts <- function(row_1, row_2){
  qry <- as.numeric(as.vector(remove_empty(df[row_1,2:ncol(df)],
                                           which="cols")))
  tmpl <- as.numeric(as.vector(remove_empty(df[row_2,2:ncol(df)],
                                            which="cols")))
  res_dtw <- dtw(qry,tmpl,keep=TRUE)
  
  # getting query names
  qry_name <- as.character(df[row_1,1])
  tmpl_name <- as.character(df[row_2,1])
  
  file_prefix <- paste(qry_name, tmpl_name, sep="_")
  
  # two way print
  file_name <- paste0(file_prefix, "_twoway.png")
  fpath <- paste0("./plots/",file_name)
  print(fpath)
  png(fpath,
      width     = 3.25,
      height    = 3.25,
      units     = "in",
      res       = 1200,
      pointsize = 2)
  heading <- paste(qry_name, "->",
                   tmpl_name, "- twoway")
  plot(res_dtw, type="twoway", main=heading)
  dev.off()

  # density print
  file_name <- paste0(file_prefix, "_density.png")
  fpath <- paste0("./plots/",file_name)
  print(fpath)
  png(fpath,
      width     = 3.25,
      height    = 3.25,
      units     = "in",
      res       = 1200,
      pointsize = 2)
  heading <- paste(qry_name, "->",
                   tmpl_name, "- density")
  plot(res_dtw, type="density", main=heading)
  dev.off()

  # threeway print
  file_name <- paste0(file_prefix, "_threeway.png")
  fpath <- paste0("./plots/",file_name)
  print(fpath)
  png(fpath,
      width     = 3.25,
      height    = 3.25,
      units     = "in",
      res       = 1200,
      pointsize = 2)
  heading <- paste(qry_name, "->",
                   tmpl_name, "- threeway")
  plot(res_dtw, type="threeway", main=heading)
  dev.off()
}

df_new <- as.data.frame(df)
mapply(FUN = dtw_viz_nearest_ts, row_1=df_dtw$qry,
                                          row_2=df_dtw$tmpl)


