import pandas as pd 
import collections
import os
import pdb

new_csv = '/data/ccusr/xinyug/lung/cura_crypto/csv/all_data_withouth_old_unlabeled.csv'
df = pd.read_csv(new_csv)
pids = set(df['PatientID'].tolist())

# all patients
print('all patients: '+str(len(pids)))

# crypto patients vs. canncer patients
crypto_num = 0
cancer_num = 0
for pid in pids:
    p_df = df[df['PatientID']==pid]
    label = p_df.iloc[0]['crypto']
    # print(label)
    if label == 1:
        crypto_num+=1
    else:
        cancer_num+=1
print('crypto patients: ' + str(crypto_num))
print('cancer patients: ' + str(cancer_num))
    
# batch patients num
tags_csv = '/data/ccusr/xinyug/lung/cura_crypto/csv/crypto_with_annt_1_2_3_contrast_731.csv'
tags3_csv = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/csv/dicom_tags/crypto_annt_tags_contrast_351_3.csv'
tags2_csv = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/csv/dicom_tags/crypto_tags_492_2.csv'
tags1_csv = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/csv/dicom_tags/crypto_tags_588_1.csv'

df3_tags = pd.read_csv(tags3_csv)
df2_tags = pd.read_csv(tags2_csv)
df1_tags = pd.read_csv(tags1_csv)

pids3_tags = set(df3_tags['PatientID'].tolist())
pids2_tags = set(df2_tags['PatientID'].tolist())
pids1_tags = set(df1_tags['PatientID'].tolist())
# print(pids3_tags)
nums3 = 0
nums2 = 0
nums1 = 0
for pid in pids:
    if pid in pids3_tags:
        nums3+=1
    elif pid in pids2_tags:
        nums2+=1
    elif pid in pids1_tags:
        nums1+=1
    # print(pid)
print('batch 1 patients num: ' + str(nums1))
print('batch 2 patients num: ' + str(nums2))
print('batch 3 patients num: ' + str(nums3))

# volume
# train_csv = '/data/ccusr/xinyug/lung/cura_crypto/csv/train.csv'
# test_csv = '/data/ccusr/xinyug/lung/cura_crypto/csv/test.csv'

# train_df = pd.read_csv(train_csv)
# test_df = pd.read_csv(test_csv)
all_data_csv = '/data/ccusr/xinyug/lung/cura_crypto/csv/all_data_withouth_old_unlabeled.csv'
df_all = pd.read_csv(all_data_csv)
seg_name_path = '/data/datasets/crypto/annt_shenzhen/nodule_seg/all/nodule_seg_names.txt'
with open(seg_name_path) as f:
    uids = [seg_name.rstrip().split('_nodule_seg')[0] for seg_name in f.readlines()]

# print(uids)
# pdb.set_trace()
def put_back_mask(df,uids):
    uids = set(uids)
    d = collections.defaultdict(list)
    for i, row in df.iterrows():
        key = str(row['PatientID']) + '_' + str(row['SliceNumber'])
        uid = os.path.split(row['img_path'])[-1].split('.nii')[0]
        d[key].append(uid)

    dd = {}
    for i, row in df.iterrows():
        cont = str(row['PatientID']) + '_' + str(row['SliceNumber'])
        key = os.path.split(row['img_path'])[-1].split('.nii')[0]
        dd[key] = cont

    u_a = {}
    for uid in uids:
        if dd.get(uid,0)==0:
            continue
        else:
            pid_num = dd[uid]
            group = d[pid_num]
            for u in group:
                u_a[u] = uid           
    
    segs = []
    pids_noseg = []
    single_nodule_path = '/data/datasets/crypto/annt_shenzhen/nodule_seg/all/single_comps'
    for i, row in df.iterrows():
        uid = os.path.split(row['img_path'])[-1].split('.nii')[0]
        if uid in u_a.keys():
            name = u_a[uid] + '_nodule_seg_single.nii.gz'
            seg_path = os.path.join(single_nodule_path,name)
            segs.append(seg_path)
        else:
            segs.append('no_seg')
            pids_noseg.append(row['PatientID'])
    df['nodule_seg_shenzhen'] = segs 
    return df , pids_noseg
    
new_df, pids_noseg = put_back_mask(df_all, uids)
# new_df.to_csv('for_nodule_volume.csv',index=False)

pids_noseg = list(set(pids_noseg))
# print(len(pids_noseg))
cnt1=cnt2=cnt3=no_group = 0
for pids_n in pids_noseg:
    # print(pids_n)
    if pids_n in pids3_tags:
        cnt3+=1
    elif pids_n in pids2_tags:
        cnt2+=1
    elif pids_n in pids1_tags:
        cnt1+=1
    else:
        no_group+=1
print('batch 1: '+str(cnt1))
print('batch 2: '+str(cnt2))
print('batch 3: '+str(cnt3))
print('no group: '+str(no_group))

# obtain convolutional kernels
def get_kernels_age_sex(df,df1_tags,df2_tags,df3_tags):
    kernels = []
    sex = []
    age = []
    m = []
    for i,row in df.iterrows():
        uid = os.path.split(row['img_path'])[-1].split('.nii')[0]
        if not df1_tags[df1_tags['SeriesInstanceUID']==uid].empty:
            kernels.append(df1_tags[df1_tags['SeriesInstanceUID']==uid].iloc[0]['ConvolutionKernel'])
            age.append(df1_tags[df1_tags['SeriesInstanceUID']==uid].iloc[0]['PatientAge'])
            sex.append(df1_tags[df1_tags['SeriesInstanceUID']==uid].iloc[0]['PatientSex'])
            m.append(df1_tags[df1_tags['SeriesInstanceUID']==uid].iloc[0]['Manufacturer'])
        elif not df2_tags[df2_tags['case']==uid].empty:
            kernels.append(df2_tags[df2_tags['case']==uid].iloc[0]['ConvolutionKernel'])
            age.append(df2_tags[df2_tags['case']==uid].iloc[0]['PatientAge'])
            sex.append(df2_tags[df2_tags['case']==uid].iloc[0]['PatientSex'])
            m.append(df2_tags[df2_tags['case']==uid].iloc[0]['Manufacturer'])
        elif not df3_tags[df3_tags['case']==uid].empty:
            kernels.append(df3_tags[df3_tags['case']==uid].iloc[0]['ConvolutionKernel'])
            age.append(df3_tags[df3_tags['case']==uid].iloc[0]['PatientAge'])
            sex.append(df3_tags[df3_tags['case']==uid].iloc[0]['PatientSex'])
            m.append(df3_tags[df3_tags['case']==uid].iloc[0]['Manufacturer'])
        else:
            print(uid)
    return kernels, sex, age, m

def get_kvp(df,df1_tags,df2_tags,df3_tags):
    kvp = []
    for i,row in df.iterrows():
        uid = os.path.split(row['img_path'])[-1].split('.nii')[0]
        if not df1_tags[df1_tags['SeriesInstanceUID']==uid].empty:
            # kernels.append(df1_tags[df1_tags['SeriesInstanceUID']==uid].iloc[0]['ConvolutionKernel'])
            kvp.append(df1_tags[df1_tags['SeriesInstanceUID']==uid].iloc[0]['KVP'])
            # sex.append(df1_tags[df1_tags['SeriesInstanceUID']==uid].iloc[0]['PatientSex'])
            # m.append(df1_tags[df1_tags['SeriesInstanceUID']==uid].iloc[0]['Manufacturer'])
        elif not df2_tags[df2_tags['case']==uid].empty:
            # kernels.append(df2_tags[df2_tags['case']==uid].iloc[0]['ConvolutionKernel'])
            kvp.append(df2_tags[df2_tags['case']==uid].iloc[0]['KVP'])
            # sex.append(df2_tags[df2_tags['case']==uid].iloc[0]['PatientSex'])
            # m.append(df2_tags[df2_tags['case']==uid].iloc[0]['Manufacturer'])
        elif not df3_tags[df3_tags['case']==uid].empty:
            # kernels.append(df3_tags[df3_tags['case']==uid].iloc[0]['ConvolutionKernel'])
            kvp.append(df3_tags[df3_tags['case']==uid].iloc[0]['KVP'])
            # sex.append(df3_tags[df3_tags['case']==uid].iloc[0]['PatientSex'])
            # m.append(df3_tags[df3_tags['case']==uid].iloc[0]['Manufacturer'])
        else:
            print(uid)
    return kvp


# kernels, sex, age, m = get_kernels_age_sex(new_df,df1_tags,df2_tags,df3_tags)
new_df = pd.read_csv('stats_tags.csv')
kvp = get_kvp(new_df,df1_tags,df2_tags,df3_tags)

# print(len(kernels))
# print(len(sex))
# print(len(age))
# print(len(m))

# new_df['ConvolutionKernel'] = kernels
# new_df['age'] = age
# new_df['sex'] = sex
# new_df['Manufacturer'] = m
new_df['kvp'] = kvp
new_df.to_csv('stats_tags.csv',index=False)
pdb.set_trace()
# 45例病患仅有1例
df_k = pd.read_csv('stats_tags.csv')
def get_info(df):
    pids = list(set(df['PatientID'].tolist()))
    
    # print(len(pids))
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    six = 0
    e=0
    sizes = []
    kernels = []
    nums_images =  []
    for pid in pids:
        cur_df = df[df['PatientID']==pid]
        cur_df = cur_df[cur_df['PatientID']==pid]
        s_num = len(set(cur_df['SliceNumber'].tolist()))
        kernel_num = len(set(cur_df['ConvolutionKernel'].tolist()))
        nums_images.append(cur_df.shape[0])
        sizes.append(s_num)
        kernels.append(kernel_num)
        # nums_images.append()
    df = pd.DataFrame({'pid':pids,'sizes':sizes,'kernels':kernels,'images':nums_images})
    return df
    # pd.to_csv('kernel_stats.csv',index=False)

        # print(cur_df.shape[0])
        # if cur_df.shape[0] == 1:
        #     one+=1
        # elif cur_df.shape[0] == 2:
        #     two+=1
        # elif cur_df.shape[0] == 3:
        #     three+=1
        # elif cur_df.shape[0] == 4:
        #     four+=1
        # elif cur_df.shape[0] == 5:
        #     five+=1
        # elif cur_df.shape[0] == 6:
        #     six+=1
        # else:
        #     e+=1
    # print(one)
    # print(two)
    # print(three)
    # print(four)
    # print(five)
    # print(six)
    # print(e)

# a_df = get_info(df_k)
# a_df.to_csv('kernel_stats.csv',index=False)
    # d = collections.defaultdict(list)
    # for i, row in df.iterrows():
        # key = str(row['PatientID']) + '_' + str(row['SliceNumber'])
        # uid = os.path.split(row['img_path'])[-1].split('.nii')[0]
        # d[key].append(uid)
b_df = pd.read_csv('kernel_stats.csv')

# one_df = b_df[b_df['sizes']==1]
# print(one_df.shape)
k_less_i = 0
k_e_i = 0
k_larger_i = 0

for i, row in b_df.iterrows():
    if row['kernels'] == row['images']:
        k_e_i+=1
    elif row['kernels'] > row['images']:
        k_larger_i+=1
    elif row['kernels'] < row['images']:
        k_less_i+=1
print(k_e_i)
print(k_larger_i)
print(k_less_i)