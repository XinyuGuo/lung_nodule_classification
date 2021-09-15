import pandas as pd
import os 
import pdb

# 1  2  3  批数据的tags
csv_file = 'crypto_with_annt_1_2_3_contrast_732.csv'
df = pd.read_csv(csv_file)
mans = set(df['Manufacturer'].tolist())
mans = list(mans)

for man in mans:
    print(man)
    man_df = df[df['Manufacturer']==man]
    # print(man_df.shape[0])
    ks = set(man_df['ConvolutionKernel'].tolist())
    # print(ks)
    for k in ks:
        print(k, man_df[man_df['ConvolutionKernel']==k].shape[0])
    pdb.set_trace()    