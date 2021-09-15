import pandas as pd 
import os  
import SimpleITK as sitk
import pdb 

def get_mask_with_largestC(mask):
    '''
    keep the largest nodule in the mask, eliminate others
    '''
    mask_arr =  sitk.GetArrayFromImage(mask)
    mask_arr[mask_arr!=0]  = 1
    new_mask = sitk.GetImageFromArray(mask_arr)
    ccIm = sitk.ConnectedComponent(new_mask)
    ccIm_ordered = sitk.RelabelComponent(ccIm)
    ccIm_ordered_arr =  sitk.GetArrayFromImage(ccIm_ordered)
    ccIm_ordered_arr[ccIm_ordered_arr>1]=0
    new_mask = sitk.GetImageFromArray(ccIm_ordered_arr)
    new_mask.CopyInformation(mask)
    return new_mask

csv_file = '../csv/crypto_case_info.csv'
csv_df = pd.read_csv(csv_file)
batch_2_file = '../csv/crypto_batch_2_names.txt'

with open(batch_2_file) as f:
    names = [name.rstrip() for name in f] 

for _,row in csv_df.iterrows():
    pid = str(row['PatientID'])
    if pid in names:
        print(pid)
        mask_path = row['mask_path']
        mask = sitk.ReadImage(mask_path)
        new_mask = get_mask_with_largestC(mask)
        new_mask_name = 'img_mask_radio.nii.gz'
        new_mask_path = os.path.join(os.path.split(mask_path)[0], new_mask_name)
        sitk.WriteImage(new_mask,new_mask_path)