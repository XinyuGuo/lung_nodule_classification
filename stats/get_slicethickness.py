import pandas as pd
import os

tags_csv = '/data/ccusr/xinyug/lung/cura_crypto/csv/crypto_with_annt_1_2_3_contrast_731.csv'
data_csv = 'stats_tags.csv'

df_tags = pd.read_csv(tags_csv)
df_data = pd.read_csv(data_csv)

cases = set(df_tags['case'].tolist())
print(len(cases))

paths = df_data['img_path'].tolist()
cases_data = [os.path.split(path)[-1].split('.nii')[0] for path in paths]

ts = []
for case in cases_data:
    if case in cases:
        s = df_tags[df_tags['case']==case]['SliceThickness'].tolist()[0]
        ts.append(s)

df_data['SliceThickness'] = ts
df_data.to_csv('stats_tags.csv', index =False)