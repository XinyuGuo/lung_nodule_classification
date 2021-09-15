import torch
from torch import nn
from torch.utils.data import DataLoader
from val_dataloader import CryptoDataset_Val

import os
import pandas as pd

import sys
sys.path.insert(1, '../../')
from model import resnet
from model import resnet_encoder
from utils.model_utils import load_checkpoint,val,getscore


def get_test_df(csv_file):
    test_df = pd.read_csv(csv_file)
    return test_df

def eval_test_single(csv_file, checkpoint_path):
    '''
    evaluate the fold 
    '''
    # load data
    print('load data ...')
    test_df = get_test_df(csv_file)
    # val_df = pd.read_csv(csv_file)
    crypto_test = CryptoDataset_Val(test_df)
    test_loader = DataLoader(crypto_test, batch_size = 16, shuffle = False, num_workers = 4)

    # load model
    print('load model ...')
    model_depth = 101
    n_classes = 1
    model = resnet.generate_model(model_depth=model_depth, n_input_channels = 1, n_classes=n_classes)
    model_eval = load_checkpoint(model, checkpoint_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_eval = model_eval.to(device)
    model_eval.eval()

    # generate results
    print('generate results ...')
    weight = torch.tensor([2.6]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight = weight)
    val_gt, val_pred , val_prob, val_loss = val(model = model_eval, criterion = criterion,\
                                                val_loader= test_loader, device = device)
    # val_gt, val_pred , val_prob, val_loss = val_dual_paths(model = model_eval, criterion = criterion,\
    #                                             val_loader= test_loader, device = device)
    
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
    df.to_csv('single_fold_' + 'test.csv', index=False)

# evaluate folds
csv_file = '/data/ccusr/xinyug/lung/cura_crypto/csv/test_4.csv'
model_path = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/results/resnet101_only_cube_gen/gen_result/checkpoint_166_.pth.tar' # 0.80
eval_test_single(csv_file, model_path)
