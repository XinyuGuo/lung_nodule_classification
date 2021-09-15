import pandas as pd
import os 
import pdb

# 1  2  3  批数据的tags
# csv_file = 'crypto_with_annt_1_2_3_contrast_731.csv'
csv_file = 'stats_tags.csv'
df = pd.read_csv(csv_file)

m = list(set(df['Manufacturer'].tolist()))

for mm in m:
    print(mm)
    new_df = df[df['Manufacturer']==mm]
    kvps = set(new_df['kvp'].tolist())
    for kvp in kvps:
        print(str(kvp) + ':'+ str(new_df[new_df['kvp']==kvp].shape[0]))
    
    df_st = new_df[new_df['SliceThickness'] == 0.8]
    print('0.8 mm')
    print(df_st.shape[0])

    df_st = new_df[new_df['SliceThickness'] == 1.0]
    print('1.0 mm')
    print(df_st.shape[0])

    df_st = new_df[new_df['SliceThickness'] == 1.25]
    print('1.25 mm')
    print(df_st.shape[0])

    df_st = new_df[new_df['SliceThickness'] > 1.25]
    print('2.0 mm')
    print(df_st.shape[0])
    pdb.set_trace()