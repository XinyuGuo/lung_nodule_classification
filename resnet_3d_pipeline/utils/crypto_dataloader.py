from torch.utils.data import Dataset
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
import pdb

class CryptoDataset_Train(DataLoader):
    '''
    crypto dataset with data augmentation
    '''
    def __init__(self, data, batch_size, num_threads_in_multithreaded=4, seed_for_shuffle=1234, return_incomplete=False,\
                 shuffle=True, infinite=True):
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,infinite)
        self.indices = list(range(len(data)))
    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data.iloc[i] for i in idx]
        nodule_data = []
        nodule_mask_data = []
        nodule_labels = []
        patient_ids = []
        for row in patients_for_batch:
            nodule_path = row['nodule_cube_80']
            # nodule_path = row['nodule_64_1']
            # nodule_path = row['nodule_path_96']
            nodule = sitk.ReadImage(nodule_path)
            nodule_arr = sitk.GetArrayFromImage(nodule)
            nodule_mask_path = row['nodule_cube_80_mask']
            # nodule_mask_path = row['mask_64_1']
            # nodule_mask_path = row['mask_path_96']
            nodule_mask = sitk.ReadImage(nodule_mask_path)
            nodule_mask_arr = sitk.GetArrayFromImage(nodule_mask)
            nodule_mask_data.append(nodule_mask_arr.astype(np.float32))
            # nodule_arr[nodule_mask_arr!=1] = -1260
            nodule_data.append(nodule_arr.astype(np.float32))
            nodule_label = row['crypto']
            nodule_labels.append(nodule_label)
            pid = str(row['PatientID'])
            patient_ids.append(pid)
        nodule_batch = np.stack(nodule_data)
        nodule_batch = np.expand_dims(nodule_batch,1)
        nodule_mask_batch = np.stack(nodule_mask_data,1)
        nodule_mask_batch =np.expand_dims(nodule_mask_batch,1) 
        nodule_batch_label = np.array(nodule_labels)
        batch_sample = {'data': nodule_batch,'seg':nodule_mask_batch,\
                        'label': nodule_batch_label.astype(np.float32),'pid':patient_ids}
        return batch_sample

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
        # nodule_path = row['nodule_64_1']
        # nodule_path = row['nodule_path_96']
        nodule_label = row['crypto']
        pid = str(row['PatientID'])
        nodule = sitk.ReadImage(nodule_path)
        nodule_arr = sitk.GetArrayFromImage(nodule)
        nodule_mask_path = row['nodule_cube_80_mask']
        # nodule_mask_path = row['mask_64_1']
        # nodule_mask_path = row['mask_path_96']
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

class CryptoDataset(Dataset):
    '''crypto dataset without data augmentation'''
    def __init__(self, crypto_df):
        # , path_root):
        self.crypto_df = crypto_df
    
    def __len__(self):
        return len(self.crypto_df)
    
    def __getitem__(self, idx):
        row = self.crypto_df.loc[idx]
        img_data = []
        pid = str(row['PatientID'])
        row = self.crypto_df.loc[idx]
        nodule_path = row['nodule_path']
        nodule_label = row['crypto']  
        nodule = sitk.ReadImage(nodule_path)
        nodule_arr = sitk.GetArrayFromImage(nodule)
        nodule_arr = self.normalize_img_arr(nodule_arr)
        img_data.append(nodule_arr.astype(np.float32))
        
        nodule_mask_path = row['nodule_mask_path']
        nodule_mask = sitk.ReadImage(nodule_mask_path)
        nodule_mask_arr = sitk.GetArrayFromImage(nodule_mask)
        lung_area_path = row['lung_area_path']
        lung_area = sitk.ReadImage(lung_area_path)
        lung_area_arr = sitk.GetArrayFromImage(lung_area) 
        lung_area_arr = self.normalize_img_arr(lung_area_arr)
        img_data.append(lung_area_arr.astype(np.float32))
        contrast = row['contrast']
        data_batch = np.stack(img_data)
        # data_batch = np.expand_dims(data_batch,1)
        # nodule_arr[nodule_mask_arr!=1] = -1260
        # nodule_arr = self.normalize_img_arr(nodule_arr)
        sample = {'data': data_batch, 'seg': nodule_mask_arr.astype(np.float32),\
                  'label': nodule_label.astype(np.float32), 'pid': pid } 
        return sample
    
    def normalize_img_arr_contrast(self, img_arr, contrast):
        '''
        normalize 
        '''
        if contrast:
            MIN_BOUND = -800  
            MAX_BOUND = 80
        else:
            MIN_BOUND = -1250    
            MAX_BOUND = 250

        img_arr = (img_arr-MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        img_arr[img_arr>1]=1
        img_arr[img_arr<0]=0
        return img_arr

    def normalize_img_arr(self, img_arr):
        '''
        normalize without contrast
        '''
        MIN_BOUND = -1000
        MAX_BOUND = 250
        img_arr = (img_arr-MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        img_arr[img_arr>1]=1
        img_arr[img_arr<0]=0
        return img_arr

class CryptoDataset_Dual_Path_Train(DataLoader):
    '''
    crypto dataset with data augmentation
    '''
    def __init__(self, data, batch_size, num_threads_in_multithreaded=4, seed_for_shuffle=1234, return_incomplete=False,\
                 shuffle=True, infinite=True):
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,infinite)
        self.indices = list(range(len(data)))
    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data.iloc[i] for i in idx]
        nodule_data = []
        nodule_mask_data = []
        dual_lung_data = []
        nodule_labels = []
        patient_ids = []
        for row in patients_for_batch:
            # nodule
            # nodule_path = row['nodule_path_80']
            nodule_path = row['nodule_cube_80']
            # nodule_path = row['nodule_64_1']
            # nodule_path = row['nodule_path_96']
            nodule = sitk.ReadImage(nodule_path)
            nodule_arr = sitk.GetArrayFromImage(nodule)
            # nodule mask
            # nodule_mask_path = row['mask_path_80']
            nodule_mask_path = row['nodule_cube_80_mask']
            # nodule_mask_path = row['mask_64_1']
            # nodule_mask_path = row['mask_path_96']
            nodule_mask = sitk.ReadImage(nodule_mask_path)
            nodule_mask_arr = sitk.GetArrayFromImage(nodule_mask)
            nodule_mask_data.append(nodule_mask_arr.astype(np.float32))
            # nodule_arr[nodule_mask_arr!=1] = -1260
            nodule_data.append(nodule_arr.astype(np.float32))
            # lung
            dual_lung_path = row['exact_lung_path']
            dual_lung = sitk.ReadImage(dual_lung_path)
            dual_lung_arr = sitk.GetArrayFromImage(dual_lung)
            dual_lung_data.append(dual_lung_arr.astype(np.float32))
            # nodule label
            nodule_label = row['crypto']
            nodule_labels.append(nodule_label)
            pid = str(row['PatientID'])
            patient_ids.append(pid)
        nodule_batch = np.stack(nodule_data)
        nodule_batch = np.expand_dims(nodule_batch,1)
        nodule_mask_batch = np.stack(nodule_mask_data,1)
        nodule_mask_batch =np.expand_dims(nodule_mask_batch,1) 
        lung_batch = np.expand_dims(dual_lung_data,1)
        nodule_batch_label = np.array(nodule_labels)
        batch_sample = {'data': nodule_batch,'seg':nodule_mask_batch,\
                        'label': nodule_batch_label.astype(np.float32),'pid':patient_ids,\
                        'dual_lung':lung_batch}
        return batch_sample

class CryptoDataset_Dual_Path_Val(Dataset):
    '''crypto dataset without data augmentation'''
    def __init__(self, crypto_df):
        # , path_root):
        self.crypto_df = crypto_df
    
    def __len__(self):
        return len(self.crypto_df)
    
    def __getitem__(self, idx):
        row = self.crypto_df.loc[idx]
        # nodule_path = row['nodule_path_80']
        nodule_path = row['nodule_cube_80']
        # nodule_path = row['nodule_64_1']
        # nodule_path = row['nodule_path_96']
        nodule_label = row['crypto']
        pid = str(row['PatientID'])
        nodule = sitk.ReadImage(nodule_path)
        nodule_arr = sitk.GetArrayFromImage(nodule)
        nodule_arr = self.normalize_img_arr(nodule_arr)

        # nodule_mask_path = row['mask_path_80']
        nodule_mask_path = row['nodule_cube_80_mask']
        # nodule_mask_path = row['mask_64_1']
        # nodule_mask_path = row['mask_path_96']
        nodule_mask = sitk.ReadImage(nodule_mask_path)
        nodule_mask_arr = sitk.GetArrayFromImage(nodule_mask)
        # nodule_arr[nodule_mask_arr!=1] = -1260
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

def get_train_transform(patch_size):
    '''
    define transforms
    '''
    transforms = []

    # scale & rotation
    transforms.append(
        SpatialTransform_2(
            patch_size=patch_size,
            # do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.5, 1.5),
            border_mode_data='constant', border_cval_data=-1250,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=0, order_data=3,
            random_crop=False,
            p_rot_per_sample=0.2, p_scale_per_sample=0.2
        )
    )    
    # mirror
    transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness
    # transforms.append(BrightnessMultiplicativeTransform((0.8, 1.2), per_channel=True, p_per_sample=0.15))

    # gamma correction 
    # transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))

    # gaussian noise
    transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15)) 

    return transforms

def get_train_transform_101(patch_size):
    '''
    define transforms
    '''
    transforms = []

    # scale & rotation
    transforms.append(
        SpatialTransform_2(
            patch_size=patch_size,
            # do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.5, 1.5),
            border_mode_data='constant', border_cval_data=-1250,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=0, order_data=3,
            random_crop=False,
            p_rot_per_sample=0.2, p_scale_per_sample=0.2
        )
    )    
    # mirror
    transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness
    transforms.append(BrightnessMultiplicativeTransform((0.8, 1.2), per_channel=True, p_per_sample=0.15))

    # gamma correction 
    transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))

    # gaussian noise
    transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15)) 

    return transforms