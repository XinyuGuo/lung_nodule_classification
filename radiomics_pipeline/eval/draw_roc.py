from sklearn.metrics import recall_score, roc_curve, auc, roc_auc_score
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

################# 
# draw ROC curve#
#################
csv_file ='/data/ccusr/xinyug/lung/cura_crypto/radiomics_pipeline/eval/radio_test.csv'
csv_df = pd.read_csv(csv_file)
gt = csv_df['gt'].tolist()
prob = csv_df['prob'].tolist()
roc_auc = roc_auc_score(gt, prob)
fpr, tpr, thresholds = roc_curve(gt, prob, pos_label=1)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='b',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc,alpha=.8)
plt.plot([0, 1], [0, 1], color='r', lw=lw, linestyle='--', label='Chance',alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('/data/ccusr/xinyug/lung/cura_crypto/radiomics_pipeline/eval/raido_test_curve.png',dpi=300,bbox_inches='tight')