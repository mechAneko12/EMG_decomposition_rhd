import pandas as pd

def mix_motion(path, list_motion, output_name, start=0):
    df = pd.DataFrame()
    list_lim = [80000,80000 + 7 + 199]
    for i, m in enumerate(list_motion):
        df_tmp = pd.read_csv(path + m, header=None)[:list_lim[i]]
        df = df.append(df_tmp, ignore_index=True)
    df.to_csv(path + output_name, header=False, index=False)
    
if __name__ == '__main__':
    list_motion =['exp0712-' + str(x) + '.csv' for x in [2,3]]
    mix_motion('dataset/20220712/', list_motion, 'exp0712_mix-2-3.csv')