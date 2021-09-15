import os
import numpy as np
import pandas as pd 
import SimpleITK as sitk 
import pdb
from data_utils import get_dual_lungs

# data_path = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/csv/crypto_largest_nodule_199_cubes_5fold.csv'
data_path = '/data/ccusr/xinyug/lung/cura_crypto/csv/crypto_all_in_study.csv'
df = pd.read_csv(data_path)
pids = df['PatientID'].tolist()
nums = df['SliceNumber'].tolist()
img_paths = df['img_path'].tolist()
lung_paths = df['lung_path'].tolist()
nodule_mask_paths = df['img_final_mask_path'].tolist()
new_size = (128,128,64)
# des_save_root = '/data/ccusr/xinyug/lung/crypto/data/crypto_all'
des_save_root = '/data/datasets/crypto/data_dev/data_annt_init_all'
# chop dual lungs
for i in range(len(img_paths)):
    key = pid[i] + '_' + str(nums[i]) 
    print(key)
    dual_lungs = get_dual_lungs(img_paths[i],lung_paths[i],new_size)
    # save iamge
    lung_name = 'dual_lungs.nii.gz'
    
    des_save_path = os.path.join(des_save_root, str(pids[i]),lung_name)
    sitk.WriteImage(dual_lungs,des_save_path)
    # pdb.set_trace()