import pandas as pd 
import os

csv_file = '/data/ccusr/xinyug/lung/cura_crypto/stats/patient_stats.csv'
df = pd.read_csv(csv_file)

pids = df['PatientID'].tolist()
age = df['age'].tolist()
sex = df['sex'].tolist()
crypto = df['crypto'].tolist()

new_crypto = []
for c in crypto:
    if c == 0:
        new_crypto.append('肺癌')
    else:
        new_crypto.append('隐球菌')

new_sex = []
for s in sex:
    if s == 'M':
        new_sex.append('男')
    else:
        new_sex.append('女')

df = pd.DataFrame({'病患编号':pids, '年龄': age, '性别': new_sex, '诊断': new_crypto})
df.to_csv('项目病患信息汇总.csv',index = False)

