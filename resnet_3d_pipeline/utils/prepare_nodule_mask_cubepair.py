import os 
import pdb
import pandas as pd 
import SimpleITK as sitk
from data_utils import get_nodule_mask_cube_pair

new_spacing = (0.68,0.68,1)
dims = (80,80,80)
cnt = 0

csv_path= '/data/ccusr/xinyug/lung/cura_crypto/csv/test_patient.csv'
df = pd.read_csv(csv_path)
nodule_paths = df['path'].tolist()
mask_paths = df['img_mask_final'].tolist()
for i in range(len(nodule_paths)):
    nodule_path = nodule_paths[i]
    print(nodule_path)
    mask_path = mask_paths[i]
    mask_name = os.path.split(mask_path)[-1]
    case = mask_name.split('_noduleSeg')[0]
    nodule, mask = get_nodule_mask_cube_pair(nodule_path, mask_path, new_spacing, dims)
    nodule_name = case + '_nodule_80.nii.gz'   
    mask_name = case + '_nodule_mask_80.nii.gz'
    nodule_save_path = os.path.join(os.path.split(nodule_path)[0],nodule_name)
    sitk.WriteImage(nodule, nodule_save_path)