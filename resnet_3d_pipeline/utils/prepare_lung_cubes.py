import os  
import numpy as np
import pandas as pd 
import SimpleITK as sitk 
import pdb
from data_utils import get_lung_cubes

data_path = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/csv/crypto_largets_nodule_info_fold_lobe.csv'
df = pd.read_csv(data_path)
pids = df['PatientID'].tolist()
img_paths = df['img_path'].tolist()
mask_paths = df['img_mask_path'].tolist()
lung_paths = df['lung_path'].tolist()
new_spacing = (1,1,0.5)

lung_indicator_csv = '../csv/crypto_resampled_lung_bbox.csv'
li_df = pd.read_csv(lung_indicator_csv)

for i in range(len(lung_paths)):
    print(pids[i])
    if pids[i] == 10305180:
        continue
    case_li_df = li_df[li_df['PatientID']==pids[i]]
    lung_indicator = case_li_df['lung_indicator'].tolist()[0]
    lung_area = get_lung_cubes(img_paths[i], mask_paths[i], new_spacing, lung_paths[i], lung_indicator)