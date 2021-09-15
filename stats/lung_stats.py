import pandas as pd 
import numpy as np 
import SimpleITK as sitk
import pdb
import math
import os

# fold_csv = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/csv/crypto_largest_nodule_199_cubes_5fold_balance.csv'
# fold_csv = 'crypto_fold_info.csv'
csv = 'patient_stats.csv'
df = pd.read_csv(csv)

# lobe_paths = []
# for i,row in df.iterrows():
#     img_path = row['img_path']
#     path_seg = os.path.split(img_path)
#     path_root = path_seg[0]
#     img_name = path_seg[-1]
#     uid = img_name.split('.nii')[0]
#     lobe_name = uid + '_lobeSeg.nii.gz'
#     lobe_path = os.path.join(path_root, lobe_name)
#     lobe_paths.append(lobe_path)
# df['lobe_path'] = lobe_paths
# df.to_csv('patient_stats.csv',index = False)

# lung_paths = df['lung_path'].tolist() 
# lobe_paths = df['lobe_path'].tolist()

# lung_vols = []
# for lung_path in lung_paths:
#     print(lung_path)
#     lung = sitk.ReadImage(lung_path)
#     spacing = lung.GetSpacing()
#     ele_vol = np.prod(spacing)
#     lung_arr = sitk.GetArrayFromImage(lung)  
#     lung_arr[lung_arr!=0] = 1
#     lung_voxels = np.sum(lung_arr)
#     lung_vol = lung_voxels * ele_vol    
#     lung_vols.append(lung_vol)

# nodule_paths = df['img_final_mask_path'].tolist()
# nodule_vols = []
# nodule_ds = []
# for nodule_path in nodule_paths:
#     print(nodule_path)
#     nodule = sitk.ReadImage(nodule_path)
#     spacing = nodule.GetSpacing()
#     ele_vol = np.prod(spacing)
#     nodule_arr = sitk.GetArrayFromImage(nodule)  
#     nodule_arr[nodule_arr>1] = 0
#     nodule_voxels = np.sum(nodule_arr)
#     nodule_vol = nodule_voxels * ele_vol    
#     nodule_vols.append(nodule_vol)

#     v = (1/math.pi)*0.75*nodule_vol
#     r = v**(1/3)
#     d = 2*r
#     nodule_ds.append(d)

# left upper 7 
# left lower 8
# right upper 4
# right middle 5
# right lower 6
lobe_paths = df['lobe_path'].tolist()
l_upper = []
l_lower = []
r_upper = []
r_mid = []
r_lower = []
for lobe_path in lobe_paths:
    print(lobe_path)
    lobe = sitk.ReadImage(lobe_path)
    spacing = lobe.GetSpacing()
    ele_vol = np.prod(spacing)
    # print(spacing)
    # print(ele_vol)
    
    # print(ele_vol)
    lobe_arr = sitk.GetArrayFromImage(lobe)
    # print((np.sum(lobe_arr[lobe_arr == 7])))
    # print(lobe_arr == 7)
    l_upper.append(np.sum([lobe_arr == 7])*ele_vol)
    # print(np.sum([lobe_arr == 7])*ele_vol)
    # pdb.set_trace()
    l_lower.append(np.sum([lobe_arr == 8])*ele_vol)
    r_upper.append(np.sum([lobe_arr == 4])*ele_vol)
    r_mid.append(np.sum([lobe_arr == 5])*ele_vol)
    r_lower.append(np.sum([lobe_arr == 6])*ele_vol)
    # print(l_upper[0])
    # print(l_lower[0])
    # print(r_upper[0])
    # print(r_mid[0])
    # print(r_lower[0])
    # print(l_upper[0]+l_lower[0]+r_upper[0]+r_mid[0]+r_lower[0])
    # pdb.set_trace()

# df['lung_volume'] = lung_vols
# df['nodule_vol'] = nodule_vols
# df['nodule_diameter'] = nodule_ds
df['left_upper'] = l_upper
df['left_lower'] = l_lower
df['right_upper'] = r_upper
df['right_mid'] = r_mid 
df['right_lower'] = r_lower
df.to_csv('patient_stats.csv', index = False)