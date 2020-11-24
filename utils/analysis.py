import os
import pandas as pd

from glob import glob

def load_df(target):
    dirs = glob(target)
    whole_df = pd.DataFrame()
    for d in dirs:
        json_path = os.path.join(d, 'train_config.json')
        csv_path = os.path.join(d, 'progress.csv')
        try:
            df_progress = pd.read_csv(csv_path, sep=' ')[-5:].mean()
            df_config = pd.read_json(json_path)[:1]
        except:
            print('Error in {}'.format(d))
            continue
        df_config['name'] = d.split('/')[-1]
        for i, j in zip(df_progress.index, df_progress.values):
            if i == 'epoch' or i == 'time':
                continue
            df_config[i] = j
        whole_df = pd.concat((whole_df, df_config))
    return whole_df

def reload_losses(df):
    before = []
    for a in ['train', 'test']:
        for b in ['angles', 'currents', 'img_reconstruction', 'img_prediction']:
            before.append(a+'/'+b)
    after = []
    for a in ['tr', 'ts']:
        for b in ['ang', 'cur', 'rec', 'prd']:
            after.append(a+'_'+b)
    for b, a in zip(before, after):
        df[a] = df[b]
    df = df.drop(before, axis=1)
    return df

def drop_unneeds(df):
    for c in df.columns:
        values_num = df[c].unique().shape[0]
        if values_num == 1:
            df = df.drop(c, axis=1)
    return df
