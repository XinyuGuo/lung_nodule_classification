import pandas as pd

new_csv = 'stats_tags.csv'
df = pd.read_csv(new_csv)

print(df.shape)
mlist = df['Manufacturer'].tolist()
mset = set(mlist)
m_ = list(mset)

for m in m_:
    print(m)
    cur_df = df[df['Manufacturer']==m]
    print('scan num: ' + str(cur_df.shape[0]))
    pids = set(cur_df['PatientID'].tolist())
    print('patient num: ' + str(len(pids)))

print(mset)
