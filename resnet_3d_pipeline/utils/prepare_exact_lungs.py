import os  
import numpy as np
import pandas as pd 
import SimpleITK as sitk 
import pdb
from data_utils import get_exact_lungs
new_size = (128,128,128)

csv_path= '/data/ccusr/xinyug/lung/cura_crypto/csv/test_patient.csv'
df = pd.read_csv(csv_path)
img_paths = df['path'].tolist()
lung_paths = df['img_lung_seg'].tolist()
nodule_paths = df['img_mask_final'].tolist()
uids = df['uid'].tolist()
for i in range(len(img_paths)):
    print(img_paths[i])
    dual_exact_lungs = get_exact_lungs(img_paths[i],lung_paths[i],nodule_paths[i],new_size)
    # save iamge
    lung_name = '_lungs.nii.gz'
    new_lung_name = uids[i] + lung_name
    des_save_path = os.path.join(os.path.split(img_paths[i])[0],new_lung_name)
    sitk.WriteImage(dual_exact_lungs,des_save_path)