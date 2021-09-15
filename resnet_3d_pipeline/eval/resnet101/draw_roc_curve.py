import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.metrics import recall_score, roc_curve, auc, roc_auc_score
from itertools import cycle

def get_gt_prob(df):
    '''
    get the fold data ground truth and probabilities
    '''
    gt = df['gt'].tolist()
    prob = df['prob'].tolist()
    return gt, prob

def get_csv_dfs():
    '''
    get the result of each fold
    '''
    csv1 = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/eval/resnet101/result_csv/dual_v2/folds_csv/dual_fold_1_val.csv' 
    csv2 = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/eval/resnet101/result_csv/dual_v2/folds_csv/dual_fold_2_val.csv' 
    csv3 = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/eval/resnet101/result_csv/dual_v2/folds_csv/dual_fold_3_val.csv' 
    csv4 = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/eval/resnet101/result_csv/dual_v2/folds_csv/dual_fold_4_val.csv' 
    csv5 = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/eval/resnet101/result_csv/dual_v2/folds_csv/dual_fold_5_val.csv' 

    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    df3 = pd.read_csv(csv3)
    df4 = pd.read_csv(csv4)
    df5 = pd.read_csv(csv5)

    dfs = [df1,df2,df3,df4,df5]
    return dfs

# script
dfs = get_csv_dfs()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
# plt.rcParams["font.family"] = "Times New Roman"
colors = ['orange','cyan','green','brown','deeppink']
lw = 1
for i, df in enumerate(dfs):
    gt, prob = get_gt_prob(df)
    fold_auc = roc_auc_score(gt, prob)
    fpr, tpr, thresholds = roc_curve(gt, prob, pos_label=1)
    ax.plot(fpr, tpr, color=colors[i], lw=lw, alpha=0.5,
            label='ROC curve of fold {0} (area = {1:0.2f})'
            ''.format(i+1, fold_auc))
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(fold_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Dual-Path ")
ax.legend(loc="lower right")

# plt.show()
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.savefig('dl_dual_roc_average.png',dpi=300,bbox_inches='tight')
