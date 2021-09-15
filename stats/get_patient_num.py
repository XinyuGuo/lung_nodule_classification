import pandas as pd 

new_csv = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/csv/dicom_tags/crypto_annt_tags_contrast_351_3.csv'
df = pd.read_csv(new_csv)
crypto_df = df[df['crypto']==1]
cry_num = len(set(crypto_df['PatientID'].tolist()))
cancer_df = df[df['crypto']==0]
cancer_num = len(set(cancer_df['PatientID'].tolist()))
print(cry_num)
print(cancer_num)

