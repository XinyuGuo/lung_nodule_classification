import os 
import pdb
import pandas as pd 
import SimpleITK as sitk
from data_utils import get_whole_lung

df = pd.read_csv('../csv/crypto_largest_nodule_199_new.csv')
img_paths = df['img_path'].tolist()
mask_paths = df['img_mask_path'].tolist()
lung_paths = df['lung_path'].tolist()
pids = df['PatientID'].tolist()
save_root = '/data/ccusr/xinyug/data/crypto/crypto_all'
new_spacing = (0.68,0.68,1)
dims = (384,384,384)
for i in range(len(img_paths)):
    pid = str(pids[i])
    print(pid)
    img_path = img_paths[i]
    mask_path = mask_paths[i]
    lung_path = lung_paths[i]
    whole_lung = get_whole_lung(img_path, lung_path, mask_path, new_spacing, dims)
    name = 'twoLungs.nii.gz'
    lung_save_path = os.path.join(save_root, pid, name)
    sitk.WriteImage(whole_lung, lung_save_path)
    pdb.set_trace()
