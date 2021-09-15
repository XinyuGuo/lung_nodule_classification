import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score,make_scorer,roc_auc_score,accuracy_score,roc_curve,auc
from sklearn import preprocessing
from sklearn.model_selection import cross_validate,cross_val_predict
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE,BorderlineSMOTE
from imblearn.pipeline import make_pipeline
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest,chi2
import pdb
import numpy as  np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time

def get_data(df,k):
    '''
    selecting the best number of features: 51
    '''
    columns = df.columns.to_list()[2:-2]
    feats_df = df[columns]
    y = feats_df['crypto_label']
    X = feats_df.drop(columns=['crypto_label'])
    X = preprocessing.MinMaxScaler().fit_transform(X)
    X = SelectKBest(chi2, k=k).fit_transform(X, y)
    return X,y

# script
# radio_feats_187.csv : num of features : 56
feats_file = '../csv/radio_feats_198_more.csv'
df = pd.read_csv(feats_file)
skf =StratifiedKFold(n_splits=10, random_state=1234, shuffle=True)
sm = SMOTE(random_state=1234,sampling_strategy='auto',k_neighbors=10)
rfc = RandomForestClassifier(n_estimators=50,random_state=1234,class_weight={1:400000,0:0.1})
clf = make_pipeline(sm,rfc)
s_set = {'specificity':make_scorer(recall_score,pos_label=0),'recall':make_scorer(recall_score),'auc':make_scorer(roc_auc_score),'acc':make_scorer(accuracy_score)}
X,y = get_data(df,21)
scores = cross_validate(clf, X, y, scoring=s_set,cv=skf)

# draw roc
t0 = time.time()
y_pred_prob = cross_val_predict(clf, X, y, cv=skf,method='predict_proba')
y_pred  = cross_val_predict(clf, X, y, cv=skf)
print('spe:' +  str(recall_score(y,y_pred,pos_label=0)))
print('sen:' +  str(recall_score(y,y_pred)))
print('auc:' +  str(roc_auc_score(y,y_pred)))
print('acc:' +  str(accuracy_score(y,y_pred)))

fpr, tpr, thresholds = roc_curve(y, y_pred_prob[:,1], pos_label=1)
font = {'family' : 'DejaVu Sans', 'size' : 10}
matplotlib.rc('font', **font)
plt.plot(fpr, tpr, lw=1)
plt.title('random forest AUC: ' + str(format(roc_auc_score(y,y_pred),'.2f')))
plt.xlabel("FPR", fontsize=10)
plt.ylabel("TPR", fontsize=10)
plt.savefig('best_so_far.png')