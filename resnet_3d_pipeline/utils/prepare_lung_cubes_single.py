import os  
import numpy as np
import pandas as pd 
import SimpleITK as sitk 
import pdb
from data_utils import get_lung_cubes

# data_path = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/csv/crypto_largets_nodule_info_fold_lobe.csv'
# df = pd.read_csv(data_path)
# pids = df['PatientID'].tolist()
# img_paths = df['img_path'].tolist()
# mask_paths = df['img_mask_path'].tolist()
# lung_paths = df['lung_path'].tolist()

img_path = '/data/ccusr/xinyug/lung/crypto/data/crypto_all/10300022/img.nii.gz'
mask_path ='/data/ccusr/xinyug/lung/crypto/data/crypto_all/10300022/img_mask.nii.gz'
lung_path ='/data/ccusr/xinyug/lung/crypto/data/crypto_all/10300022/lungSeg.nii.gz'
new_spacing = (0.68,0.68,1)
lung_indicator_csv = '../csv/crypto_resampled_lung_bbox.csv'
li_df = pd.read_csv(lung_indicator_csv)
case_li_df = li_df[li_df['PatientID']==10300022]
lung_indicator = case_li_df['lung_indicator'].tolist()[0]
# print(lung_indicator)
# pdb.set_trace()
lung_area = get_lung_cubes(img_path, mask_path, new_spacing, lung_path, lung_indicator)
