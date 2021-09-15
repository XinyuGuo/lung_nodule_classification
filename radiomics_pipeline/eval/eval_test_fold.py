import _pickle as cPickle
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

def load_model(model_path):
    with open(model_path, 'rb') as fid:
        clf = cPickle.load(fid)
    return clf 

def get_val_test(train_path, test_path,fold):
    df_train = pd.read_csv(train_path) 
    df_test = pd.read_csv(test_path)
    df_val = df_train[df_train['fold'] == fold]
    y_val = df_val['crypto_label']
    y_test = df_test['crypto_label']
    columns = df_train.columns.to_list()[2:-4]
    X_val = df_val[columns]    
    X_test = df_test[columns]
    return X_val, y_val, X_test, y_test

def get_val_test_auc(model,X_val,y_val,X_test,y_test):
    pred_val = model.predict_proba(X_val)
    pred_test =model.predict_proba(X_test)
    auc_val = roc_auc_score(y_val, pred_val[:,1])
    auc_test = roc_auc_score(y_test,pred_test[:,1] )
    return auc_val, auc_test, pred_val[:,1], pred_test[:,1]

def save_to_file(pred_val, y_val, pred_test, y_test, val_path, test_path):
    pred_val_df = pd.DataFrame({'gt':y_val,'prob':pred_val})
    pred_test_df = pd.DataFrame({'gt':y_test,'prob':pred_test})
    pred_val_df.to_csv(val_path, index=False)
    pred_test_df.to_csv(test_path,index=False)

# data prepare
fold = 5
train_path = '../csv/radio_feats_train_4.csv'
test_path = '../csv/radio_feats_test_4.csv'
X_val, y_val, X_test, y_test = get_val_test(train_path, test_path,fold)

# load model
model_path = '/data/ccusr/xinyug/lung/cura_crypto/radiomics_pipeline/eval/radio_fold_5.pkl'
model = load_model(model_path)

# calculate auc
auc_val, auc_test, pred_val, pred_test = get_val_test_auc(model,X_val,y_val,X_test,y_test)
print('auc val: ' + str(auc_val) + '\n' + 'auc test: ' + str(auc_test))

# save csv file for drawing AUC curve
root = '/data/ccusr/xinyug/lung/cura_crypto/radiomics_pipeline/eval/'
val_name = 'fold_' + str(fold) + '_final_val.csv'
val_save_path = os.path.join(root,val_name)
test_name = 'fold_' + str(fold) + '_final_test.csv'
test_save_path = os.path.join(root,test_name)
save_to_file(pred_val, y_val, pred_test, y_test, val_save_path, test_save_path)