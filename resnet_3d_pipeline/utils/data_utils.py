import SimpleITK as sitk
import numpy as np
import os
import pandas as pd 
import math
from tqdm import tqdm
import sys
import pdb
import shutil
import torch
sys.path.insert(1, '../model')
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform

# ****************** functions : chop cubes ********************** #
def padding(img, mask):
    '''
    if cube dimension is not (64,64,64), the do the padding
    '''
    img_arr = sitk.GetArrayFromImage(img)
    mask_arr = sitk.GetArrayFromImage(mask)
    i_size = img_arr.shape
    extra = np.array((64,64,64)) - np.array(i_size)
    new_img_arr = np.pad(img_arr,
                        ((extra[0]//2,extra[0]-extra[0]//2),\
                         (extra[1]//2,extra[1]-extra[1]//2),\
                         (extra[2]//2,extra[2]-extra[2]//2)),\
                         'edge')
    new_mask_arr = np.pad(mask_arr,
                        ((extra[0]//2,extra[0]-extra[0]//2),\
                         (extra[1]//2,extra[1]-extra[1]//2),\
                         (extra[2]//2,extra[2]-extra[2]//2)),\
                         'edge')
    
    new_mask = sitk.GetImageFromArray(new_mask_arr)
    new_img = sitk.GetImageFromArray(new_img_arr)
    new_img.SetSpacing((0.68,0.68,1))
    new_mask.SetSpacing((0.68,0.68,1))
    return new_img, new_mask

def chopCT(img, mask, nodule_num, save_root, case):
    '''
    chop nodule and mask cube
    '''
    mask_arr = sitk.GetArrayFromImage(mask)
    mask_arr[mask_arr>0] = 1
    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(img)
    mask_comp = sitk.RelabelComponent(sitk.ConnectedComponent(mask))
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(mask_comp)

    for i in range(1,nodule_num+1):
        bbox = np.array(shape_filter.GetBoundingBox(i))
        center = bbox[0:3] + bbox[3:6]//2
        nodule = img[center[0]-32:center[0]+32,center[1]-32:center[1]+32,center[2]-32:center[2]+32]
        nodule_mask = mask[center[0]-32:center[0]+32,center[1]-32:center[1]+32,center[2]-32:center[2]+32]
        if nodule_mask.GetSize() != (80,80,80):
            nodule, nodule_mask = padding(nodule, nodule_mask)

        nodule_name = case + '_nodule_80.nii.gz'
        mask_name = case + '_nodule_mask_80.nii.gz'
        nodule_path = os.path.join(save_root, nodule_name)
        nodule_mask_path = os.path.join(save_root, mask_name)
        sitk.WriteImage(nodule, nodule_path)
        sitk.WriteImage(nodule_mask, nodule_mask_path)

def resampleCT(img_info, new_spacing):
    '''
    resampleCT the original image based on the new_spacing
    '''
    if type(img_info) == str:
        img_path = img_info    
        img = sitk.ReadImage(img_path)
    else:
        img = img_info
    spacing = np.array(img.GetSpacing())
    size = np.array(img.GetSize())
    new_size = spacing * size * 1./new_spacing
    r_filter = sitk.ResampleImageFilter()
    r_filter.SetOutputDirection(img.GetDirection())
    r_filter.SetOutputOrigin(img.GetOrigin())    
    r_filter.SetOutputSpacing(new_spacing)
    new_size = (int(new_size[0]),int(new_size[1]),int(new_size[2])) 
    r_filter.SetSize(new_size)
    new_img = r_filter.Execute(img)
    return new_img

def get_nodule_mask_cubes(img_path, mask_path, n_num, new_spacing, case):
    '''
    get the nodules and the corresponding masks for training the network
    ''' 
    # step 1: resample the original image and its mask
    new_img = resampleCT(img_path, new_spacing)
    new_mask = resampleCT(mask_path, new_spacing)

    # step 2: chop the nodule cube and its mask cube
    save_root = os.path.split(mask_path)[0]
    chopCT(new_img, new_mask, n_num, save_root, case)

# ****************** functions : get lung cubes ************************* #
def get_object_bbox(lung):
    '''
    get the bbox of one lung
    '''
    mask_comp = sitk.RelabelComponent(sitk.ConnectedComponent(lung))
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(mask_comp)
    bbox = np.array(shape_filter.GetBoundingBox(1))
    return bbox 

def paddingLung(lung_area,dims):
    '''
    padding lung to the dims
    '''
    lung_arr = sitk.GetArrayFromImage(lung_area)
    i_size = lung_arr.shape
    extra = np.array(dims) - np.array(i_size)
    new_lung_arr = np.pad(lung_arr,
                        ((extra[0]//2,extra[0]-extra[0]//2),\
                         (extra[1]//2,extra[1]-extra[1]//2),\
                         (extra[2]//2,extra[2]-extra[2]//2)),\
                         'edge')
    
    new_lung = sitk.GetImageFromArray(new_lung_arr)
    new_lung.SetSpacing((0.68,0.68,1))
    new_lung.SetSpacing((0.68,0.68,1))
    return new_lung

def chopLung(img,bbox,dims,nodule_mask,lung_mask):
    # chopLung(new_img,lung_bbox,dims,new_mask,new_lung,nodule_lung)
    '''
    chop lung area from the image
    '''
    # refine the lung mask
    lung_arr = sitk.GetArrayFromImage(lung_mask)
    lung_arr[lung_arr!=1] = 0
    nodule_arr = sitk.GetArrayFromImage(nodule_mask)
    lung_arr[nodule_arr==1] = 1
    new_lung_mask = sitk.GetImageFromArray(lung_arr)
    new_lung_mask = sitk.BinaryFillhole(sitk.BinaryDilate(sitk.GetImageFromArray(lung_arr),3))
    new_lung_mask.CopyInformation(lung_mask)
    
    # chop lung area and lung mask by bbox, then padding
    hx,hy,hz = dims[0]//2, dims[1]//2, dims[2]//2
    c_x,c_y,c_z = bbox[0] + bbox[3]//2,bbox[1] + bbox[4]//2, bbox[2] + bbox[5]//2
    size = new_lung_mask.GetSize()
    x_start = max(0,c_x-hx) 
    x_end = min(c_x+hx,size[0])
    y_start = max(0,c_y-hx)
    y_end = min(c_y+hy,size[1])
    z_start = max(0,c_z-hz)
    z_end = min(c_z+hz,size[2])
    lung_area = img[x_start:x_end, y_start:y_end, z_start:z_end]
    cur_lung_mask = new_lung_mask[x_start:x_end, y_start:y_end, z_start:z_end]

    # mask out the lung area 
    if lung_area.GetSize() != dims:
        new_lung_area = paddingLung(lung_area, dims)
        new_cur_lung_mask = paddingLung(cur_lung_mask, dims)
        new_lung_area_arr = sitk.GetArrayFromImage(new_lung_area)
        new_cur_lung_mask_arr = sitk.GetArrayFromImage(new_cur_lung_mask)
        new_lung_area_arr[new_cur_lung_mask_arr!=1] = -1250
        new_lung_area_2 = sitk.GetImageFromArray(new_lung_area_arr)
        new_lung_area_2.CopyInformation(new_lung_area)
        return new_lung_area_2
    else:
        lung_area_arr = sitk.GetArrayFromImage(lung_area)
        cur_lung_mask_arr = sitk.GetArrayFromImage(cur_lung_mask)
        lung_area_arr[cur_lung_mask_arr!=1] = -1250
        new_lung_area = sitk.GetImageFromArray(lung_area_arr)
        new_lung_area.CopyInformation(lung_area)
        return new_lung_area

    return new_lung_area

def downSample(lung_area):
    '''
    downsmaple lung to 64*64*64
    '''
    new_spacing = (2.72,2.72,4.0)
    new_size = (64,64,64)
    r_filter = sitk.ResampleImageFilter()
    r_filter.SetOutputDirection(lung_area.GetDirection())
    r_filter.SetOutputOrigin(lung_area.GetOrigin())    
    r_filter.SetOutputSpacing(new_spacing)
    r_filter.SetSize(new_size)
    new_img = r_filter.Execute(lung_area)
    return new_img

def change_lung_mask(lung_path,indicator):
    '''
    only keep the nodule lung
    '''
    lung_mask = sitk.ReadImage(lung_path)
    lung_arr = sitk.GetArrayFromImage(lung_mask)
    lung_arr[lung_arr!=indicator] = 0
    lung_arr[lung_arr==indicator] = 1
    new_lung_mask = sitk.GetImageFromArray(lung_arr)
    new_lung_mask.CopyInformation(lung_mask)
    return new_lung_mask

def get_lung_cubes(img_path, mask_path, new_spacing, lung_path, indicator):
    '''
    img_path: lung image path, mask_path: nodule mask path, lung path: lung mask path
    indicator: which lung the nodule belongs to
    get the lung image containing the nodule
    '''
    # step 1: resample the original image and its mask
    new_img = resampleCT(img_path, new_spacing)
    new_mask = resampleCT(mask_path, new_spacing)
    new_lung_mask = change_lung_mask(lung_path, indicator)
    new_lung = resampleCT(new_lung_mask, new_spacing)

    # step 2: get the bounding box of the lung
    bbox = get_object_bbox(new_lung)

    # step 3: chop lung area 
    dims = (256,256,256)
    lung_area = chopLung(new_img,bbox,dims,new_mask,new_lung)

    # step 4: downsmaple
    final_lung = downSample(lung_area)
    
    # step 5: save image
    save_path = os.path.join(os.path.split(lung_path)[0],'lungArea.nii.gz')
    sitk.WriteImage(final_lung,save_path)

# ***************** functions : Chop the nodule-mask pair **************************** #
def paddingNodule(nodule,dims):
    '''
    padding image
    '''
    nodule_arr = sitk.GetArrayFromImage(nodule)
    i_size = nodule_arr.shape
    extra = np.array(dims) - np.array(i_size)
    new_nodule_arr = np.pad(nodule_arr,
                        ((extra[0]//2,extra[0]-extra[0]//2),\
                         (extra[1]//2,extra[1]-extra[1]//2),\
                         (extra[2]//2,extra[2]-extra[2]//2)),\
                         'edge')
    
    new_nodule = sitk.GetImageFromArray(new_nodule_arr)
    new_nodule.SetSpacing((0.68,0.68,1))
    new_nodule.SetSpacing((0.68,0.68,1))
    return new_nodule
    
def chopPair(img, nodule_mask, dims):
    '''
    get one pair of nodule and mask's cube pair
    '''
    mask_comp = sitk.ConnectedComponent(nodule_mask)
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(mask_comp)
    bbox = np.array(shape_filter.GetBoundingBox(1))

    hx,hy,hz = dims[0]//2, dims[1]//2, dims[2]//2
    c_x,c_y,c_z = bbox[0] + bbox[3]//2,bbox[1] + bbox[4]//2, bbox[2] + bbox[5]//2
    size = nodule_mask.GetSize()
    x_start = max(0,c_x-hx) 
    x_end = min(c_x+hx,size[0])
    y_start = max(0,c_y-hx)
    y_end = min(c_y+hy,size[1])
    z_start = max(0,c_z-hz)
    z_end = min(c_z+hz,size[2])
    nodule_area = img[x_start:x_end, y_start:y_end, z_start:z_end]
    nodule_mask_area = nodule_mask[x_start:x_end, y_start:y_end, z_start:z_end]

    # padding 
    if nodule_area.GetSize() != dims:
        nodule_area = paddingNodule(nodule_area, dims)
        nodule_mask_area = paddingNodule(nodule_mask_area, dims)
        
    return nodule_area, nodule_mask_area

def get_nodule_mask_cube_pair(img_path, mask_path, new_spacing, dims):
    '''
    get the nodules and the corresponding masks for training the network
    ''' 
    # step 1: resample the original image and its mask
    new_img = resampleCT(img_path, new_spacing)
    new_mask = resampleCT(mask_path, new_spacing)

    # step 2: chop the nodule cube and its mask cube
    nodule, mask = chopPair(new_img, new_mask, dims)
    return nodule, mask

# ****************** functions : chop two lungs ************************ #
def paddingWholeLung(wholeLung,dims,is_mask=False):
    '''
    padding image
    '''
    whole_arr = sitk.GetArrayFromImage(wholeLung)
    i_size = whole_arr.shape
    print(whole_arr.shape)
    pdb.set_trace()
    extra = np.array(dims) - np.array(i_size)
    if is_mask:
        padding_method = 'constant'
    else:
        padding_method = 'edge'
    new_whole_arr = np.pad(whole_arr,
                        ((extra[0]//2,extra[0]-extra[0]//2),\
                         (extra[1]//2,extra[1]-extra[1]//2),\
                         (extra[2]//2,extra[2]-extra[2]//2)),\
                         padding_method)
    
    new_whole = sitk.GetImageFromArray(new_whole_arr)
    new_whole.SetSpacing(wholeLung.GetSpacing())
    new_whole.SetOrigin(wholeLung.GetOrigin())
    new_whole.SetDirection(wholeLung.GetDirection())
    return new_whole

def chopWholeLung(img, lung_mask, nodule_mask, lb, ub, dims):
    '''
    chop the area containing two lungs
    '''
    # chop the lung area
    x_start = lb[0]
    x_end = ub[0]
    y_start = lb[1]
    y_end = ub[1]
    z_start = lb[2]
    z_end = ub[2]
    lung_area = img[x_start:x_end, y_start:y_end, z_start:z_end]
    lung_mask_area = lung_mask[x_start:x_end, y_start:y_end, z_start:z_end]
    nodule_mask_area = nodule_mask[x_start:x_end, y_start:y_end, z_start:z_end]
    
    # padding 
    lung_area = paddingWholeLung(lung_area, dims)
    lung_mask_area = paddingWholeLung(lung_mask_area, dims, True)
    nodule_mask_area = paddingWholeLung(nodule_mask_area, dims, True)
    
    # refine the lung mask and mask out the non-lung area
    l_arr = sitk.GetArrayFromImage(lung_mask_area)
    l_arr[l_arr>=2] = 1
    n_arr = sitk.GetArrayFromImage(nodule_mask_area)
    la_arr = sitk.GetArrayFromImage(lung_area)
    l_arr[n_arr==1] = 1
    lung_mask_2 = sitk.BinaryFillhole(sitk.BinaryDilate(sitk.GetImageFromArray(l_arr),3))
    l_arr_2 = sitk.GetArrayFromImage(lung_mask_2)
    la_arr[l_arr_2!=1] = -1250
    final_lung_area = sitk.GetImageFromArray(la_arr)
    final_lung_area.CopyInformation(lung_area)
    return final_lung_area

def get_single_lung_bbox(lung):
    # connected_filter = sitk.ConnectedComponentImageFilter()
    connected_filter = sitk.RelabelComponentImageFilter()
    mask_comp = connected_filter.Execute(lung)
    # obj_nums = connected_filter.GetObjectCount()
    # print(obj_nums)
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(mask_comp)
    bbox = np.array(shape_filter.GetBoundingBox(1))
    return bbox 

def get_3d_bbox(lung_mask):
    '''
    get the whole bounding box of two lungs
    '''
    # get right bbox
    lung_arr = sitk.GetArrayFromImage(lung_mask)
    _lung_arr = lung_arr.copy()
    _lung_arr[lung_arr==2] = 1
    _lung_arr[lung_arr==3] = 0
    right_lung = sitk.GetImageFromArray(_lung_arr)
    right_lung.CopyInformation(lung_mask)
    right_bbox = get_single_lung_bbox(right_lung)
    # get left bbox
    lung_arr[lung_arr==2] = 0
    lung_arr[lung_arr==3] = 1
    left_lung = sitk.GetImageFromArray(lung_arr)
    left_lung.CopyInformation(lung_mask)
    left_bbox = get_single_lung_bbox(left_lung)
    # get the whole bbox
    r_l = right_bbox[0:3]
    r_u = right_bbox[0:3] + right_bbox[3:6]
    l_l = left_bbox[0:3]
    l_u = left_bbox[0:3] + left_bbox[3:6]
    new_l = np.array([r_l,l_l])
    lower = np.min(new_l,axis=0)
    new_u = np.array([r_u,l_u])
    upper = np.max(new_u,axis=0)
    return lower, upper

# ****************** functions : data augmentation ********************** #
def get_train_transform():
    '''
    define transforms
    '''
    transforms = []
    # scale & rotation
    transforms.append(
        SpatialTransform_2(
            patch_size=(64,64,64),
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=0, order_data=3,
            random_crop=False,
            p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )  
    # mirror
    transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness
    transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))

    # gamma correction 
    transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))

    # gaussian noise
    transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15)) 

    return transforms

def get_offaug_gen(data_csv, aug_batch_size, path_root):
    # path_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    df = pd.read_csv(data_csv)
    train_df = df[df['fold_1'] == 0]
    crypto_df = train_df[train_df['crypto']==1]
    nodule_df = train_df[train_df['crypto']==0]
    crypto_num = crypto_df.shape[0]
    aug_times = math.ceil(nodule_df.shape[0]/crypto_num)   
    crypto_df.reset_index(drop=True, inplace=True)
    crypto_df = make_paths(crypto_df, path_root)
    offaug_loader = CryptoDataset_1_c_Aug(crypto_df, batch_size = aug_batch_size)
    all_transforms = Compose(get_train_transform())
    offaug_gen = MultiThreadedAugmenter(offaug_loader, all_transforms, num_processes=1,\
                                   num_cached_per_queue=1,\
                                   seeds=None, pin_memory=True)
    return offaug_gen, aug_times, crypto_num 

def gen_offaug_data(offaug_gen, aug_times, aug_batch_num, temp_path):
    labels = []
    for aug_time in range(aug_times):
        print(aug_time)
        for batch_num in range(aug_batch_num):
            aug_batch = offaug_gen.next()
            save_aug_data(aug_batch, aug_time, batch_num, temp_path)

def save_aug_data(aug_batch, aug_time, batch_num, temp_path):
    data = np.squeeze(aug_batch['data'],1)
    pids = aug_batch['pids']
    d_num = data.shape[0]
    for i in range(d_num):
        img = sitk.GetImageFromArray(data[i])
        img.SetOrigin((0.0,0.0,0.0))
        img.SetSpacing((0.68, 0.68, 1.0))
        img_name = pids[i] + '_aug_' + str(aug_time) + '_' + str(batch_num) + '_' + str(i) + '.nii.gz'
        img_path = os.path.join(temp_path,img_name)
        sitk.WriteImage(img, img_path)

def get_balanced_train_df(data_csv, crypto_aug_temp_path, path_root):
    aug_files = os.listdir(crypto_aug_temp_path)
    aug_paths = [os.path.join(crypto_aug_temp_path, aug_file) for aug_file in aug_files]
    aug_names = [name.split('.nii')[0] for name in aug_files]
    df = pd.read_csv(data_csv)
    fold_train_df = df[df['fold_1']==0]
    fold_train_df.reset_index(drop=True, inplace=True)
    for index, row in fold_train_df.iterrows():
        part_path = row['nodule_path']
        fold_train_df.at[index, 'nodule_path'] = os.path.join(path_root, part_path)
    
    pids = fold_train_df['PatientID'].tolist()
    paths = fold_train_df['nodule_path'].tolist()
    new_pids = pids + aug_names
    new_paths = paths + aug_paths
    labels = fold_train_df['crypto'].tolist()
    new_labels = labels + len(aug_names) * [1]
    balanced_dict = {'PatientID':new_pids, 'nodule_path': new_paths, 'crypto': new_labels}
    balanced_df = pd.DataFrame(balanced_dict)
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)    
    balanced_df.to_csv('./csv/balanced_nodule_train.csv',index=False)
    return balanced_df

def create_new_dataset(**kwargs):
    # paras loaded 
    crypto_aug_temp_path = kwargs['crypto_aug_temp_path']
    data_csv = kwargs['data_csv'] 
    aug_batch_size = kwargs['aug_batch_size'] 
    path_root = kwargs['path_root']
    # build the dir holding the augmented data
    if not os.path.exists(crypto_aug_temp_path):
        os.mkdir(crypto_aug_temp_path)
    else:
        shutil.rmtree(crypto_aug_temp_path)
        os.mkdir(crypto_aug_temp_path)    
    # generate the augmented data and the new dataframe for training 
    offaug_gen, aug_times, crypto_num = get_offaug_gen(data_csv, aug_batch_size, path_root)
    aug_batch_num = math.ceil(crypto_num/aug_batch_size)
    gen_offaug_data(offaug_gen, aug_times, aug_batch_num, crypto_aug_temp_path)
    balanced_df = get_balanced_train_df(data_csv,crypto_aug_temp_path,path_root)
    return balanced_df

def get_offaug_gen_with_mask(data_csv, aug_batch_size, path_root):
    # path_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    df = pd.read_csv(data_csv)
    train_df = df[df['fold_1'] == 0]
    crypto_df = train_df[train_df['crypto']==1]
    nodule_df = train_df[train_df['crypto']==0]
    crypto_num = crypto_df.shape[0]
    aug_times = math.ceil(nodule_df.shape[0]/crypto_num)   
    crypto_df.reset_index(drop=True, inplace=True)
    crypto_df = make_paths(crypto_df, path_root)
    offaug_loader = CryptoDataset_1_c_Aug_Mask_and_Nodule(crypto_df, batch_size = aug_batch_size)
    all_transforms = Compose(get_train_transform())
    offaug_gen = MultiThreadedAugmenter(offaug_loader, all_transforms, num_processes=1,\
                                   num_cached_per_queue=1,\
                                   seeds=None, pin_memory=True)
    return offaug_gen, aug_times, crypto_num 

def gen_offaug_data_mask(offaug_gen, aug_times, aug_batch_num, temp_path):
    labels = []
    for aug_time in range(aug_times):
        print(aug_time)
        for batch_num in range(aug_batch_num):
            aug_batch = offaug_gen.next()
            save_aug_data_mask(aug_batch, aug_time, batch_num, temp_path)

def save_aug_data_mask(aug_batch, aug_time, batch_num, temp_path):
    data = np.squeeze(aug_batch['data'],1)
    mask = np.squeeze(aug_batch['seg'],1)
    pids = aug_batch['pids']
    d_num = data.shape[0]
    for i in range(d_num):
        img = sitk.GetImageFromArray(data[i])
        mask = sitk.GetImageFromArray(mask[i])
        img.SetOrigin((0.0,0.0,0.0))
        img.SetSpacing((0.68, 0.68, 1.0))
        mask.SetOrigin((0.0,0.0,0.0))
        mask.SetSpacing((0.68, 0.68, 1.0))
        img_name = pids[i] + '_aug_' + str(aug_time) + '_' + str(batch_num) + '_' + str(i) + '.nii.gz'
        mask_name = pids[i] + '_aug_' + str(aug_time) + '_' + str(batch_num) + '_' + str(i) + '_mask' + '.nii.gz'
        img_path = os.path.join(temp_path,img_name)
        mask_path = os.path.join(temp_path, mask_name)
        sitk.WriteImage(img, img_path)
        sitk.WriteImage(mask, mask_path)

def create_new_dataset_with_mask(**kwargs):
    # paras loaded 
    crypto_aug_temp_path = kwargs['crypto_aug_temp_path']
    data_csv = kwargs['data_csv'] 
    aug_batch_size = kwargs['aug_batch_size'] 
    path_root = kwargs['path_root']
    # build the dir holding the augmented data
    if not os.path.exists(crypto_aug_temp_path):
        os.mkdir(crypto_aug_temp_path)
    else:
        shutil.rmtree(crypto_aug_temp_path)
        os.mkdir(crypto_aug_temp_path)    
    # generate the augmented data and the new dataframe for training 
    offaug_gen, aug_times, crypto_num = get_offaug_gen_with_mask(data_csv, aug_batch_size, path_root)
    aug_batch_num = math.ceil(crypto_num/aug_batch_size)
    gen_offaug_data_mask(offaug_gen, aug_times, aug_batch_num, crypto_aug_temp_path)
    balanced_df = get_balanced_train_df(data_csv,crypto_aug_temp_path,path_root)
    return balanced_df

def train_classifer(**kwargs):
    '''
    extract 3d resnet features
    '''
    res_encoder = kwargs['encoder']
    model.eval()
    train_loader = kwargs['img_loader']
    device = kwargs['device']
    pids = []
    image_feats = []
    with torch.no_grad():
        for train_batch in train_loader:
            batch_pids = train_batch['pid']
            train_data = train_batch['data'].to(device)
            train_data = torch.unsqueeze(train_data,1)
            train_label = train_batch['label'].to(device)
            _, output = res_encoder(train_data)
            
            print(output.shape)
# ***************** functions : Chop doubel lungs **************************** #
def resize(img, new_size):
    '''
    img: target image
    new_size: new lung size
    '''
    size = np.array(img.GetSize())
    spacing = np.array(img.GetSpacing())
    new_spacing = spacing * size * 1./ np.array(new_size)
    r_filter = sitk.ResampleImageFilter()
    r_filter.SetOutputDirection(img.GetDirection())
    r_filter.SetOutputOrigin(img.GetOrigin())    
    r_filter.SetOutputSpacing(new_spacing)
    r_filter.SetSize(new_size)
    new_img = r_filter.Execute(img)
    return new_img

def get_dual_lungs(img_path, lung_mask_path, nodule_mask_path, lung_cube_size):
    '''
    img_path: the path of the ct image
    lung_mask_path: the path of the lung mask
    nodule_mask_path: the path of the nodule_mask
    lung_cube_szie: 64*64*128
    '''
    # step 1 : refine the lung mask 
    lung_i = sitk.ReadImage(lung_mask_path)
    lung_a = sitk.GetArrayFromImage(lung_i)
    lung_a[lung_a>0] = 1
    nodule_a = sitk.GetArrayFromImage(sitk.ReadImage(nodule_mask_path))
    lung_a[nodule_a == 1] = 1
    new_lung_i = sitk.BinaryFillhole(sitk.BinaryDilate(sitk.GetImageFromArray(lung_a),3))
    
    # step 2 : mask out dual lungs
    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    new_lung_a = sitk.GetArrayFromImage(new_lung_i)
    img_arr[new_lung_a != 1] = -2000
    new_img = sitk.GetImageFromArray(img_arr)
    new_img.CopyInformation(lung_i)

    # step 3 : resize
    dual_lungs =resize(new_img, lung_cube_size)
    
    return dual_lungs

# ***************** functions : Chop exact double lungs **************************** #
def get_3d_bbox(lung_mask):
    # get right bbox
    lung_arr = sitk.GetArrayFromImage(lung_mask)
    _lung_arr = lung_arr.copy()
    _lung_arr[lung_arr==2] = 1
    _lung_arr[lung_arr==3] = 0
    right_lung = sitk.GetImageFromArray(_lung_arr)
    right_lung.CopyInformation(lung_mask)
    right_bbox = get_single_lung_bbox(right_lung)
    # get left bbox
    lung_arr[lung_arr==2] = 0
    lung_arr[lung_arr==3] = 1
    left_lung = sitk.GetImageFromArray(lung_arr)
    left_lung.CopyInformation(lung_mask)
    left_bbox = get_single_lung_bbox(left_lung)
    # get the whole bbox
    r_l = right_bbox[0:3]
    r_u = right_bbox[0:3] + right_bbox[3:6]
    l_l = left_bbox[0:3]
    l_u = left_bbox[0:3] + left_bbox[3:6]
    new_l = np.array([r_l,l_l])
    lower = np.min(new_l,axis=0)
    new_u = np.array([r_u,l_u])
    upper = np.max(new_u,axis=0)
    return lower, upper

def get_lung_area(img, lower, upper):
    lung_area = img[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]
    return lung_area

def resize_img(img, new_size):
    spacing = np.array(img.GetSpacing())
    size = np.array(img.GetSize())
    new_spacing = spacing * size * 1./np.array(new_size)
    r_filter = sitk.ResampleImageFilter()
    r_filter.SetOutputDirection(img.GetDirection())
    r_filter.SetOutputOrigin(img.GetOrigin())    
    r_filter.SetOutputSpacing(tuple(new_spacing))
    r_filter.SetSize(new_size)
    new_img = r_filter.Execute(img)
    return new_img

def get_pure_lungs(lung_area, area_mask):
    '''
    mask out the normal lung area
    '''
    mask_arr = sitk.GetArrayFromImage(area_mask)
    mask_arr[mask_arr==2] = 1
    mask_arr[mask_arr==3] = 1
    lung_arr = sitk.GetArrayFromImage(lung_area)
    lung_arr[mask_arr==0] = 0
    pure_lungs = sitk.GetImageFromArray(lung_arr)
    pure_lungs.CopyInformation(lung_area)
    return pure_lungs

def refine(img_mask, lung_mask):
    '''
    refine the lung mask
    '''
    lung_mask = sitk.BinaryFillhole(sitk.BinaryDilate(lung_mask,3))
    lung_mask_arr = sitk.GetArrayFromImage(lung_mask)
    lung_mask_arr[lung_mask_arr == 2] = 1
    lung_mask_arr[lung_mask_arr == 3] = 1
    img_mask_arr = sitk.GetArrayFromImage(img_mask)
    lung_mask_arr[img_mask_arr==1] = 1
    refined_lm = sitk.GetImageFromArray(lung_mask_arr)
    refined_lm.CopyInformation(lung_mask)
    return refined_lm

def get_exact_lungs(ct_path, lung_mask_path, img_mask_path, new_size):
    '''
    chop the area containing two exact lungs, get rid of the abdomen part
    '''
    # step 1 : load the image and the mask
    img = sitk.ReadImage(ct_path)
    lung_mask = sitk.ReadImage(lung_mask_path)
    img_mask = sitk.ReadImage(img_mask_path)
    # step 2 : refine the lung mask
    lung_mask = refine(img_mask,lung_mask)  
    # step 3 : get the bounding box of two lungs
    lower, upper = get_3d_bbox(lung_mask)
    # step 3 : get the lung area based on the bbox 
    lung_area = get_lung_area(img, lower, upper)
    area_mask = get_lung_area(lung_mask, lower, upper)
    # step 4 : mask out the lung area
    pure_lungs = get_pure_lungs(lung_area, area_mask)
    # step 5 : resize the lung area
    new_lung_area = resize_img(pure_lungs, new_size)
    return new_lung_area