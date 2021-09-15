import os 
import pandas as pd
import numpy as np 
import SimpleITK as sitk
import shutil
import pdb
import csv
from multiprocessing import Pool
import datetime
from data_utils import get_nodule_mask_cubes

# load data paths 
# df = pd.read_csv('../csv/crypto_largest_nodule_199_cube_5fold.csv')
df = pd.read_csv('/data/ccusr/xinyug/lung/cura_crypto/csv/crypto_all_in_study.csv')
pids = df['PatientID'].tolist()
cases = df['case'].tolist()
img_paths = df['img_path'].tolist()
mask_paths = df['img_mask_path'].tolist()
# n_nums = df['Crypto_Num'].tolist()
new_spacing = (0.68,0.68,1)

# chop cubes
for i in range(len(img_paths)):
    print(pids[i])
    # get_nodule_mask_cubes(img_paths[i],mask_paths[i],n_nums[i],new_spacing)
    get_nodule_mask_cubes(img_paths[i],mask_paths[i],1,new_spacing,cases[i])
    pdb.set_trace()