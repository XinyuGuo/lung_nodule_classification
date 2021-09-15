import torch
from torch import nn
from torch.utils.data import DataLoader
from val_dataloader import CryptoDataset_Val

import os
import pandas as pd 

import sys
sys.path.insert(1, '../../')
from model import resnet
from utils.model_utils import load_checkpoint,val,getscore

def get_val_df(csv_file, fold_name):
    '''
    get the validation fold
    '''
    df = pd.read_csv(csv_file)
    val_df = df[df[fold_name] == 1]
    val_df.reset_index(drop=True, inplace=True)
    return val_df

def get_val_df(csv_file,fold):
    df = pd.read_csv(csv_file)
    val_df = df[df['fold'] == fold]
    val_df.reset_index(drop=True, inplace=True)
    return val_df

def eval_fold(csv_file, fold, checkpoint_path):
    '''
    evaluate the fold 
    '''
    # print(fold_name)
    # load data
    val_df = get_val_df(csv_file, fold_name)
    crypto_val = CryptoDataset_Val(val_df)
    val_loader = DataLoader(crypto_val, batch_size = 16, shuffle = False, num_workers = 4)

    fold_name = 'fold_' + str(fold)
    print(fold_name)
    # load data
    print('load data ...')
    val_df = get_val_df(csv_file, fold)
    crypto_val = CryptoDataset_Val(val_df)
    val_loader = DataLoader(crypto_val, batch_size = 16, shuffle = False, num_workers = 4)

    # load model
    print('load model ...')
    model_depth = 101
    n_classes = 1
    model = resnet.generate_model(model_depth=model_depth, n_input_channels = 1, n_classes=n_classes)
    model_eval = load_checkpoint(model, checkpoint_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_eval = model_eval.to(device)

    # generate results
    print('generate result ...')
    weight = torch.tensor([2.15]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight = weight)
    val_gt, val_pred , val_prob, val_loss = val(model = model_eval, criterion = criterion,\
                                                val_loader= val_loader, device = device)

    gts = []
    for gt in val_gt:
        gts.append(gt)
    
    probs=[]
    for prob in val_prob:
        probs.append(prob)

    spe, sen, auc = getscore(val_gt, val_pred, val_prob) 
    spe = round(spe, 2)
    sen = round(sen, 2)
    auc = round(auc, 2)
    print('spe ' + str(spe) + ', sen ' + str(sen) + ', auc ' + str(auc) + ', val loss ' + str(val_loss))
    df = pd.DataFrame({'gt':gts, 'prob':probs})
    df.to_csv('single_fold_' + str(fold) + '_val.csv', index=False)

# evaluate folds
csv_file = '/data/ccusr/xinyug/lung/cura_crypto/csv/train_4.csv'

# fold 1
model_1_path = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/results/resnet101_only_cube/fold_2/checkpoint_89_.pth.tar'
eval_fold(csv_file, 1, model_1_path)

# fold 2
model_2_path = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/results/resnet101_only_cube_gen/gen_result/checkpoint_121_.pth.tar'
eval_fold(csv_file, 2, model_2_path)

# fold 3
model_3_path = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/results/resnet101_only_cube_gen/gen_result/checkpoint_166_.pth.tar'
eval_fold(csv_file, 3, model_3_path)

# fold 4 
model_4_path = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/results/resnet101_only_cube/fold_2/checkpoint_95_.pth.tar'
eval_fold(csv_file, 4, model_4_path)

# fold 5
model_5_path = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/results/resnet101_only_cube/fold_2/checkpoint_155_.pth.tar'
eval_fold(csv_file, 5, model_5_path)