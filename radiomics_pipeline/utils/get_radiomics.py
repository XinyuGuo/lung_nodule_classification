from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
oj = os.path.join
from multiprocessing import Pool
from scipy import ndimage
from tqdm import tqdm
import datetime
import pdb
import csv

import pandas as pd
# compute first order features
def get_first_order_feats(img, mask, settings):
    firstOrderFeatures = firstorder.RadiomicsFirstOrder(img, mask, **settings)
    firstOrderFeatures.enableAllFeatures()
    results_dict = firstOrderFeatures.execute()
    renamed = {}
    for k in results_dict.keys():
        renamed['firstorder_' + k] = results_dict[k]
    return renamed
# compute shape features
def get_shape_feats(img, mask, settings):
    shapeFeatures = shape.RadiomicsShape(img, mask, **settings)
    shapeFeatures.enableAllFeatures()
    results_dict = shapeFeatures.execute()
    return results_dict
# compute GLCM features
def get_GLCM_feats(img, mask, settings):
    glcmFeatures = glcm.RadiomicsGLCM(img, mask, **settings)
    glcmFeatures.enableAllFeatures()
    results_dict = glcmFeatures.execute()
    renamed = {}
    for k in results_dict.keys():
        renamed['glcm_' + k] = results_dict[k]    
    return renamed
# compute GLRLM features
def get_GLRLM_feats(img, mask, settings):
    glrlmFeatures = glrlm.RadiomicsGLRLM(img, mask, **settings)
    glrlmFeatures.enableAllFeatures()
    results_dict = glrlmFeatures.execute()
    renamed = {}
    for k in results_dict.keys():
        renamed['glrlm_' + k] = results_dict[k]    
    return renamed
# compute GLSZM features
def get_GLSZM_feats(img, mask, settings):
    glszmFeatures = glszm.RadiomicsGLSZM(img, mask, **settings)
    glszmFeatures.enableAllFeatures()
    results_dict = glszmFeatures.execute()
    renamed = {}
    for k in results_dict.keys():
        renamed['glszm_' + k] = results_dict[k]    
    return renamed
# compute features after applying Lap of Gaussian filter
def get_first_order_LoG_feats(img, mask, sigmas, settings, compute_GLCM=True, compute_GLRLM=True, compute_GLSZM=False):
    results_dict = {}
    for logImage, imageTypename, inputSettings in imageoperations.getLoGImage(img, mask, sigma=sigmas):
        firstorder_dict = get_first_order_feats(logImage, mask, settings)
        for k in firstorder_dict.keys():
            results_dict[imageTypename + '_' + k] = firstorder_dict[k]
        if compute_GLCM:
            GLCM_dict = get_GLCM_feats(logImage, mask, settings)
            for k in GLCM_dict.keys():
                results_dict[imageTypename + '_' + k] = GLCM_dict[k]
        if compute_GLRLM:
            GLRLM_dict = get_GLRLM_feats(logImage, mask, settings)
            for k in GLRLM_dict.keys():
                results_dict[imageTypename + '_' + k] = GLRLM_dict[k]
        if compute_GLSZM:
            GLSZM_dict = get_GLSZM_feats(logImage, mask, settings)
            for k in GLSZM_dict.keys():
                results_dict[imageTypename + '_' + k] = GLSZM_dict[k]
    return results_dict
# compute features after applying Wavelet of Gaussian filter
def get_first_order_wavelet_feats(img, mask, settings, compute_GLCM=True, compute_GLRLM=True, compute_GLSZM=False):
    results_dict = {}
    for decompositionImage, decompositionName, inputSettings in imageoperations.getWaveletImage(img, mask):
        print(decompositionName)
        firstorder_dict = get_first_order_feats(decompositionImage, mask, settings)
        for k in firstorder_dict.keys():
            results_dict[decompositionName + '_' + k] = firstorder_dict[k]
        if compute_GLCM:
            GLCM_dict = get_GLCM_feats(decompositionImage, mask, settings)
            for k in GLCM_dict.keys():
                results_dict[decompositionName + '_' + k] = GLCM_dict[k]
        if compute_GLRLM:
            GLRLM_dict = get_GLRLM_feats(decompositionImage, mask, settings)
            for k in GLRLM_dict.keys():
                results_dict[decompositionName + '_' + k] = GLRLM_dict[k]
        if compute_GLSZM:
            GLSZM_dict = get_GLSZM_feats(decompositionImage, mask, settings)
            for k in GLSZM_dict.keys():
                results_dict[decompositionName + '_' + k] = GLSZM_dict[k]
    return results_dict
# threshold the original image
def window(img):
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    origin = img.GetOrigin()
    img = sitk.GetArrayFromImage(img)
    img[img <= 0] = 0
    img[img >= 200] = 200
    img = sitk.GetImageFromArray(img)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    return img

# augmentation
def augmentation(img, mask, generator):
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    origin = img.GetOrigin()
    img = sitk.GetArrayFromImage(img)
    mask = sitk.GetArrayFromImage(mask)
    shape = img.shape

    batch_mask = []
    batch_x = []
    max_int = np.iinfo(np.int32).max
    sd = np.random.randint(max_int)
    for s in range(img.shape[0]):
        x = img[s, :]
        x = np.reshape(x, [x.shape[0], x.shape[1], 1])
        x = generator.random_transform(x.astype(np.float32), seed=sd)
        x = generator.standardize(x)
        batch_x.append(x)

        m = mask[s, :]
        m = np.reshape(m, [m.shape[0], m.shape[1], 1])
        m = generator.random_transform(m, seed=sd)
        m = generator.standardize(m)
        batch_mask.append(m)
        
    batch_x = np.asarray(batch_x)
    batch_mask = np.asarray(batch_mask)
    batch_x = batch_x.reshape(shape)
    batch_mask = batch_mask.reshape(shape)

    batch_x = sitk.GetImageFromArray(batch_x)
    batch_x.SetSpacing(spacing)
    batch_x.SetOrigin(origin)
    batch_x.SetDirection(direction)    
    batch_mask = sitk.GetImageFromArray(batch_mask)
    batch_mask.SetSpacing(spacing)
    batch_mask.SetOrigin(origin)
    batch_mask.SetDirection(direction)

    return batch_x, batch_mask

def run(pair):
    img_path = pair[0]
    mask_path = pair[1]
    crypto_label = pair[2]
    error = False
    pid = img_path.split('/')[-2]
    print('patient:', pid)
    try:
        i = 0    
        img = sitk.ReadImage(img_path)
        itk_mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(itk_mask)
        label_set = np.unique(mask)
        for label in label_set:
            if not label == 0:
                settings = {'binWidth': 30, 'interpolator': sitk.sitkBSpline, 'resampledPixelSpacing': None, 'label': label}
                print('settings:', settings)
                feats = {}
                sigmas = [10]
                feats.update(get_first_order_feats(img, itk_mask, settings))
                feats.update(get_shape_feats(img, itk_mask, settings))
                feats.update(get_GLCM_feats(img, itk_mask, settings))
                feats.update(get_GLRLM_feats(img, itk_mask, settings))
                feats.update(get_GLSZM_feats(img, itk_mask, settings))
                # feats.update(get_first_order_LoG_feats(img, mask, sigmas, settings))
                # feats.update(get_first_order_wavelet_feats(img, itk_mask, settings))
                for k in feats.keys():
                    feats[k] = float(feats[k])
                feats['crypto_label'] = crypto_label
                feats['label'] = label
                columns = ['IMAGE_PATH', 'MASK_PATH'] + list(feats.keys())
                if i == 0:
                    feat_dt = pd.DataFrame(columns=columns)
                feats['IMAGE_PATH'] = img_path
                feats['MASK_PATH'] = mask_path
                feats['CASE_ID'] = pid
                feat_dt = feat_dt.append(feats, ignore_index=True)
                i += 1
        return [feat_dt,pid,error]
    except RuntimeError:
        error = True
        return [feat_dt,pid,error]
        
def gather_feats_dfs(pair):
    df = pair[0]
    case_id = pair[1]
    error = pair[2]
    if not error:
        features_dfs.append(df)
    else:
        csv_write_error.writerow([case_id])


if __name__ == '__main__':    
    #### =============== feature extraction =============== ####

    # get paths ready: img + mask
    # csv_df = '../csv/crypto_largest_nodule_199_new.csv'
    csv_df = '/data/ccusr/xinyug/lung/cura_crypto/csv/test_4.csv'
    sus_df = pd.read_csv(csv_df)
    imgs_info =  zip(sus_df['img_path'].tolist(),sus_df['img_final_mask_path'].tolist(),sus_df['crypto'])
    path_root = os.path.abspath(os.getcwd())
    infos = [(oj(path_root,img_info[0]), oj(path_root,img_info[1]),img_info[2]) for img_info in imgs_info ]
    # infos = infos[0:2]
    # 0 - 8
    # infos = infos[174:199]
 
    # define error log
    error_log = '../csv/error_log.csv'
    elog = open(error_log,'w')
    csv_write_error = csv.writer(elog)

    print('start extracting features ...')
    features_dfs = []
    p = Pool(8)
    e1 = datetime.datetime.now()
    for i in range(len(infos)):
        p.apply_async(run, (infos[i],), callback=gather_feats_dfs)
    
    p.close()
    p.join()    
    e2 = datetime.datetime.now()

    whole_features_df = pd.concat(features_dfs)
    whole_features_df.to_csv('../csv/radio_feats_test_4.csv',index=False)
    print((e2-e1))