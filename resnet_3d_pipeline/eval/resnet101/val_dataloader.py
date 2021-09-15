from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
import os

class CryptoDataset_Val(Dataset):
    '''crypto dataset without data augmentation'''
    def __init__(self, crypto_df):
        # , path_root):
        self.crypto_df = crypto_df
    
    def __len__(self):
        return len(self.crypto_df)
    
    def __getitem__(self, idx):
        row = self.crypto_df.loc[idx]
        nodule_path = row['nodule_cube_80']
        nodule_label = row['crypto']
        pid = str(row['PatientID'])
        nodule = sitk.ReadImage(nodule_path)
        nodule_arr = sitk.GetArrayFromImage(nodule)
        nodule_mask_path = row['nodule_cube_80_mask']
        nodule_mask = sitk.ReadImage(nodule_mask_path)
        nodule_mask_arr = sitk.GetArrayFromImage(nodule_mask)
        # nodule_arr[nodule_mask_arr!=1] = -1260
        nodule_arr = self.normalize_img_arr(nodule_arr)
        sample = {'data': nodule_arr.astype(np.float32), 'seg': nodule_mask_arr.astype(np.float32),\
                  'label': nodule_label.astype(np.float32), 'pid': pid } 
        return sample
    
    def normalize_img_arr(self, img_arr):
        '''
        normalize 
        '''
        MIN_BOUND = -800  
        MAX_BOUND = 80
        img_arr = (img_arr-MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        img_arr[img_arr>1]=1
        img_arr[img_arr<0]=0
        return img_arr


class CryptoDataset_Dual_Path_Val(Dataset):
    '''crypto dataset without data augmentation'''
    def __init__(self, crypto_df):
        # , path_root):
        self.crypto_df = crypto_df
    
    def __len__(self):
        return len(self.crypto_df)
    
    def __getitem__(self, idx):
        row = self.crypto_df.loc[idx]
        nodule_path = row['nodule_cube_80']
        nodule_label = row['crypto']
        pid = str(row['PatientID'])
        # pid = '000000'
        nodule = sitk.ReadImage(nodule_path)
        nodule_arr = sitk.GetArrayFromImage(nodule)
        nodule_arr = self.normalize_img_arr(nodule_arr)
        nodule_mask_path = row['nodule_cube_80_mask']
        nodule_mask = sitk.ReadImage(nodule_mask_path)
        nodule_mask_arr = sitk.GetArrayFromImage(nodule_mask)
        dual_lung_path = row['exact_lung_path']
        dual_lung = sitk.ReadImage(dual_lung_path)
        dual_lung_arr = sitk.GetArrayFromImage(dual_lung)
        dual_lung_arr = self.normalize_img_arr(dual_lung_arr)
        
        sample = {'data': nodule_arr.astype(np.float32), 'seg': nodule_mask_arr.astype(np.float32),\
                  'label': nodule_label.astype(np.float32), 'pid': pid,\
                  'dual_lung':dual_lung_arr.astype(np.float32)} 
        return sample

    def normalize_img_arr(self, img_arr):
        '''
        normalize 
        '''
        MIN_BOUND = -1000 
        MAX_BOUND = 200
        img_arr = (img_arr-MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        img_arr[img_arr>1]=1
        img_arr[img_arr<0]=0
        return img_arr