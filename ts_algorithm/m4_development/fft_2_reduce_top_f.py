import pandas as pd
from tqdm import tqdm

def check_type(s: pd.Series) -> pd.Series:
    if float(s["PSD"]):
        return 1
    else:
        return 0
    
    
    
def main():
    print("starting computation of top 5 frequencies per ts")
    df = pd.read_csv("../../data/df_apx_win_m4.csv") 
    # print("running iteration to find the issue")
    # df["test"] = df.apply(check_type, axis=1)
    # print(df[df["test"]==0])
    df["PSD"] = df["PSD"].astype(float)
    df["freq_apx_idx"] = df["freq_apx_idx"].astype(int)
    print("df read with shape: {}".format(df.shape))

    df_res = pd.DataFrame(columns=['ts_name', 'freq_ids'])
    for ts in tqdm(df.groupby(["type", "ts_name"])):
        # ts_l.append(ts)

        N = 5
        ts_type = ts[0][0]
        ts_name = ts[0][1]
        df_sub = ts[1]
    
        df_sub['freq_apx_idx'] = df_sub['freq_apx_idx'].astype(int)

        freq_l = df_sub['freq_apx_idx'].tolist()
        PSD_l = df_sub['PSD'].tolist()

        # returns largest index positions sorted from smallest to biggest
        idx_powerful_PSD = sorted(range(len(PSD_l)), key= lambda
                                  x: PSD_l[x])[-N:]

    
        # get the largest values by their index into the list
        freq_idx = [freq_l[i] for i in idx_powerful_PSD]
        if ts_name == "H364":
            print(f"processing {ts_name} - ts_type: {ts_type}\n")
            print(freq_idx)
    
        df_res = df_res.append(pd.DataFrame({'ts_name': ts_name,
                                             'type': ts_type,
                                             'freq_ids': str(freq_idx)},
                                            index=[0]))
 
        
    df_res.reset_index(inplace=False)
    df_res.to_csv("../../data/df_freq_l_m4.csv", index=False)


if __name__=="__main__":
    main()
