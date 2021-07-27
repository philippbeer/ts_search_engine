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
df_dtw <- df_dtw_cols %>% purrr::map_dfc(setNames, object = list(integer()))

# looping to find closest series for each series in all others
for (qry_row in 1:2) {
  # empty tibble for results
  qry <- as.numeric(as.vector(remove_empty(df[qry_row,2:no_cols],
                                           which="cols")))
  
  tmp_shortest_dist <- 0
  
  for (tmpl_row in 1:15) {
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
  
  print("+++++++++ Closest Connection ++++++++++")
  print(paste("Row", qry_row, "is closest to", tmp_row))
  print("+++++++++++++++++++++++++++++++++++++++")
  df_dtw <- df_dtw %>% add_row(tibble_row(qry = qry_row, tmpl = tmp_row))
}

# compute final results and print them
for (row in 1:nrow(df_dtw)) {
  print(row)
  qry <- as.numeric(as.vector(remove_empty(df[row$qry,2:no_cols],
                                           which="cols")))
  tmpl <- as.numeric(as.vector(remove_empty(df[row$tmpl,2:no_cols],
                                            which="cols")))
  two_text <- paste()
  tmp_dtw <- dtw(qry,tmp,keep=TRUE)
  qry_name <- as.character(df[row$qry,1])
  qry_name <- as.character(df[row$qry,1])
  p1 <- plot(tmp_dtw, type="twoway")
  p2 <- plot(tmp_dtw, type="density")
  p3 <- plot(tmp_dtw, type="threeway")
  par(mfrows=c(2,2))
  
}


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
  png(filename=fpath, res=300)
  plot(res_dtw, type="twoway")
  
  # density print
  file_name <- paste0(file_prefix, "_density.png")
  fpath <- paste0("./plots/",file_name)
  png(filename=fpath, res=300)
  plot(res_dtw, type="density")
  
  # threeway print
  file_name <- paste0(file_prefix, "_threeway.png")
  fpath <- paste0("./plots/",file_name)
  png(filename=fpath, res=300)
  plot(res_dtw, type="threeway")  
 
  dev.off()
  
}

df_new <- as.data.frame(df)
mapply(FUN = dtw_viz_nearest_ts, row_1=df_dtw$qry,
                                          row_2=df_dtw$tmpl)


