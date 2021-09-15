import pandas as pd
import pdb
import numpy as np 
import scipy.stats

# df = pd.read_csv('stats_tags.csv')
# pids = list(set(df['PatientID'].tolist()))

# dfs = []
# for pid in pids:
#     cur_df = df[df['PatientID']==pid]
#     dfs.append(cur_df.iloc[[0]])

# new_df = pd.concat(dfs)
# # new_df.to_csv('patient_stats.csv',index=False)


# crypto_ages = df_crypto['age'].tolist()
# cancer_ages = df_cancer['age'].tolist()
# crypto_ages = np.array([int(age[0:-1]) for age in crypto_ages])
# cancer_ages = np.array([int(age[0:-1]) for age in cancer_ages])

# crypto_mean = np.mean(crypto_ages)
# crypto_std = np.std(crypto_ages)
# cancer_mean = np.mean(cancer_ages)
# cancer_std = np.std(cancer_ages)

# print('crypto age mean: ' + str(crypto_mean))
# print('crypto age std: ' + str(crypto_std))
# print('cancer age mean: ' + str(cancer_mean))
# print('cancer age std: ' + str(cancer_std))
# print(scipy.stats.ttest_ind(cancer_ages,crypto_ages,equal_var = False))

# ages = df['age'].tolist()
# ages = np.array([int(age[0:-1]) for age in ages])
# print(np.mean(ages),np.std(ages))

# males =df_crypto[df_crypto['sex']=='M']
# females =df_crypto[df_crypto['sex']=='F']
# print('crypto:' + str(len(males)) + ' ' + str(len(females)))

# males =df_cancer[df_cancer['sex']=='M']
# females =df_cancer[df_cancer['sex']=='F']
# print('cancer:' + str(len(males)) + ' ' + str(len(females)))
# print(df_cancer.shape)

# crypto_lung = np.array(df_crypto['lung_volume'].tolist())
# cancer_lung = np.array(df_cancer['lung_volume'].tolist())
# print('crypto lung mean: ' + str(np.mean(crypto_lung)))
# print('crypto lung std: ' + str(np.std(crypto_lung)))
# print('cancer lung mean: ' + str(np.mean(cancer_lung)))
# print('cancer lung std: ' + str(np.std(cancer_lung)))
# print(scipy.stats.ttest_ind(cancer_lung,crypto_lung,equal_var = False))

# lung = np.array(df['lung_volume'].tolist())
# print(np.mean(lung))
# print(np.std(lung))

df = pd.read_csv('patient_stats.csv')
# df = df[df['nodule_vol']>2000]
df_crypto = df[df['crypto']==1]
df_cancer = df[df['crypto']==0]
# crypto_n = np.array(df_crypto['nodule_vol'].tolist())
# cancer_n = np.array(df_cancer['nodule_vol'].tolist())

# print('crypto n_vol mean: ' + str(np.mean(crypto_n)))
# print('crypto n_vol std: ' + str(np.std(crypto_n)))
# print('cancer n_vol mean: ' + str(np.mean(cancer_n)))
# print('cancer n_val std: ' + str(np.std(cancer_n)))
# print(scipy.stats.ttest_ind(cancer_n,crypto_n,equal_var = False))

# nodule = np.array(df['nodule_vol'].tolist())
# print(np.mean(nodule))
# print(np.std(nodule))

# 
# crypto_n = np.array(df_crypto['nodule_diameter'].tolist())
# cancer_n = np.array(df_cancer['nodule_diameter'].tolist())

# print('crypto n_diameter mean: ' + str(np.mean(crypto_n)))
# print('crypto n_diameter std: ' + str(np.std(crypto_n)))
# print('cancer n_diameter mean: ' + str(np.mean(cancer_n)))
# print('cancer n_diameter std: ' + str(np.std(cancer_n)))
# print(scipy.stats.ttest_ind(cancer_n,crypto_n,equal_var = False))

# nodule = np.array(df['nodule_diameter'].tolist())
# print(np.mean(nodule))
# print(np.std(nodule))


# df['nodule_vol'] = nodule_vols
# df['nodule_diameter'] = nodule_ds

# （（（（（（
# crypto_lu = np.array(df_crypto['left_upper'].tolist())
# cancer_lu = np.array(df_cancer['left_upper'].tolist())

# print('crypto n_diameter mean: ' + str(np.mean(crypto_lu)))
# print('crypto n_diameter std: ' + str(np.std(crypto_lu)))
# print('cancer n_diameter mean: ' + str(np.mean(cancer_lu)))
# print('cancer n_diameter std: ' + str(np.std(cancer_lu)))
# print(scipy.stats.ttest_ind(cancer_lu,crypto_lu,equal_var = False))

# lu = np.array(df['left_upper'].tolist())
# print(np.mean(lu))
# print(np.std(lu))

# crypto_lo = np.array(df_crypto['left_lower'].tolist())
# cancer_lo = np.array(df_cancer['left_lower'].tolist())

# print('crypto n_diameter mean: ' + str(np.mean(crypto_lo)))
# print('crypto n_diameter std: ' + str(np.std(crypto_lo)))
# print('cancer n_diameter mean: ' + str(np.mean(cancer_lo)))
# print('cancer n_diameter std: ' + str(np.std(cancer_lo)))
# print(scipy.stats.ttest_ind(cancer_lo,crypto_lo,equal_var = False))

# lo = np.array(df['left_lower'].tolist())
# print(np.mean(lo))
# print(np.std(lo))

# crypto_ru = np.array(df_crypto['right_upper'].tolist())
# cancer_ru = np.array(df_cancer['right_upper'].tolist())

# print('crypto n_diameter mean: ' + str(np.mean(crypto_ru)))
# print('crypto n_diameter std: ' + str(np.std(crypto_ru)))
# print('cancer n_diameter mean: ' + str(np.mean(cancer_ru)))
# print('cancer n_diameter std: ' + str(np.std(cancer_ru)))
# print(scipy.stats.ttest_ind(cancer_ru,crypto_ru,equal_var = False))

# ru = np.array(df['right_upper'].tolist())
# print(np.mean(ru))
# print(np.std(ru))

# crypto_rm = np.array(df_crypto['right_mid'].tolist())
# cancer_rm = np.array(df_cancer['right_mid'].tolist())

# print('crypto n_diameter mean: ' + str(np.mean(crypto_rm)))
# print('crypto n_diameter std: ' + str(np.std(crypto_rm)))
# print('cancer n_diameter mean: ' + str(np.mean(cancer_rm)))
# print('cancer n_diameter std: ' + str(np.std(cancer_rm)))
# print(scipy.stats.ttest_ind(cancer_rm,crypto_rm,equal_var = False))

# rm = np.array(df['right_mid'].tolist())
# print(np.mean(rm))
# print(np.std(rm))


crypto_rl = np.array(df_crypto['right_lower'].tolist())
cancer_rl = np.array(df_cancer['right_lower'].tolist())

print('crypto n_diameter mean: ' + str(np.mean(crypto_rl)))
print('crypto n_diameter std: ' + str(np.std(crypto_rl)))
print('cancer n_diameter mean: ' + str(np.mean(cancer_rl)))
print('cancer n_diameter std: ' + str(np.std(cancer_rl)))
print(scipy.stats.ttest_ind(cancer_rl,crypto_rl,equal_var = False))

rl = np.array(df['right_lower'].tolist())
print(np.mean(rl))
print(np.std(rl))