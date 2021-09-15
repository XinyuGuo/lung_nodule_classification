import torch
from torch import nn
from torch.utils.data import DataLoader
from val_dataloader import CryptoDataset_Dual_Path_Val

import os
import pandas as pd

import sys
sys.path.insert(1, '../../')
from model import resnet
from model.dual_resnet_cls import dual_resnet_cls_eval
from model import resnet_encoder
from utils.model_utils import load_checkpoint,val_dual_paths,getscore


def get_test_df(csv_file):
    test_df = pd.read_csv(csv_file)
    return test_df

def eval_test_dual(csv_file, checkpoint_path):
    '''
    evaluate the fold 
    '''
    # load data
    print('load data ...')
    test_df = get_test_df(csv_file)
    # val_df = pd.read_csv(csv_file)
    crypto_test = CryptoDataset_Dual_Path_Val(test_df)
    test_loader = DataLoader(crypto_test, batch_size = 8, shuffle = False, num_workers = 4)

    # load model
    print('load model ...')
    model_depth = 101
    n_classes = 1
    encoder_nodule = resnet_encoder.generate_model(model_depth=model_depth, n_input_channels = 1, n_classes=n_classes) # n_class = 1039
    encoder_lung = resnet_encoder.generate_model(model_depth=model_depth, n_input_channels = 1, n_classes=n_classes)
    pretrain_model_path = checkpoint_path

    model_eval = dual_resnet_cls_eval(encoder_nodule,encoder_lung)
    model_params = torch.load(checkpoint_path)
    model_eval.load_state_dict(model_params['state_dict'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_eval = model_eval.to(device)
    model_eval.eval()

    # generate results
    print('generate results ...')
    weight = torch.tensor([2.6]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight = weight)
    # val_gt, val_pred , val_prob, val_loss = val(model = model_eval, criterion = criterion,\
                                                # val_loader= val_loader, device = device)
    val_gt, val_pred , val_prob, val_loss = val_dual_paths(model = model_eval, criterion = criterion,\
                                                val_loader= test_loader, device = device)
    gts = []
    for gt in val_gt:
        gts.append(gt)
        # print(gt)
    print('***********')
    
    probs=[]
    for prob in val_prob:
        probs.append(prob)
        # print(prob)

    spe, sen, auc = getscore(val_gt, val_pred, val_prob) 
    spe = round(spe, 2)
    sen = round(sen, 2)
    auc = round(auc, 2)
    print('spe ' + str(spe) + ', sen ' + str(sen) + ', auc ' + str(auc) + ', val loss ' + str(val_loss))
    df = pd.DataFrame({'gt':gts, 'prob':probs})
    df.to_csv('dual_fold_' + 'test.csv', index=False)


# evaluate folds
csv_file = '/data/ccusr/xinyug/lung/cura_crypto/csv/test_4.csv'
model_path = '/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/results/resnet101_gen_results/val_result_3/checkpoint_14_.pth.tar' # 0
eval_test_dual(csv_file, model_path)
