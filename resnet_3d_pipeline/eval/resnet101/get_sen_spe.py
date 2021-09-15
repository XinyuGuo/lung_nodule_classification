import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix

dual_val_csv = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/eval/resnet101/result_csv/dual_v2/dual_fold_2_val.csv'
dual_test_csv = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/eval/resnet101/result_csv/dual_v2/dual_fold_test.csv'

single_val_csv = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/eval/resnet101/result_csv/single_v2/single_fold_3_val.csv'
single_test_csv = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/eval/resnet101/result_csv/single_v2/single_fold_test.csv'

radio_val_csv = '/data/ccusr/xinyug/lung/cura_crypto/radiomics_pipeline/eval/radio_val.csv'
radio_test_csv = '/data/ccusr/xinyug/lung/cura_crypto/radiomics_pipeline/eval/radio_test.csv'

df_dual_val = pd.read_csv(dual_val_csv)
df_dual_test = pd.read_csv(dual_test_csv)

df_single_val = pd.read_csv(single_val_csv)
df_single_test = pd.read_csv(single_test_csv)

df_radio_val = pd.read_csv(radio_val_csv)
df_radio_test = pd.read_csv(radio_test_csv)

def get_sen_spe(df,th):
    gt = np.array(df['gt'].tolist())
    prob  = np.array(df['prob'].tolist())
    # print(prob)
    prob[prob>th] = 1
    prob[prob<=th] = 0
    # print(gt)
    # print(prob)
    tn, fp, fn, tp = confusion_matrix(gt, prob).ravel()
    specificity = tn / (tn+fp)
    sencitivity = tp / (tp+fn)
    print(sencitivity)
    print(specificity)
    
get_sen_spe(df_dual_val,0.39)    # 0.85/0.85/0.39
get_sen_spe(df_dual_test,0.38)   # 0.76/0.74/0.38 0.78/0.71/0.375 # 0.80/0.69/0.37

get_sen_spe(df_single_val,0.525) # 0.82/0.70/0.525 # 0.74/0.75/0.56 
get_sen_spe(df_single_test,0.58) # 0.73 0.72/0.58 # 0.80/0.71/0.57 

get_sen_spe(df_radio_val,0.35)   # 0.80/0.75/0.35
get_sen_spe(df_radio_test,0.27)  # 0.70/0.69/0.27