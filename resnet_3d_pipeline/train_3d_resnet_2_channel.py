import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import Compose
import math 
import os
import pandas as pd
import time
import pdb
# import self-defined functions
from model import resnet
from utils.model_utils import load_pretrained_model_2_c, freezelayers, train_noaug, val_noaug, getscore, save_results
from utils.crypto_dataloader import CryptoDataset_Train, CryptoDataset_Val, get_train_transform,CryptoDataset

# prepare the data entry
print('prepare the augment loader ...')
# csv_file = './csv/crypto_largest_nodule_fold_198.csv'
csv_file = './csv/crypto_largest_nodule_199_cubes_5fold.csv'
df = pd.read_csv(csv_file)
fold_name = 'fold_1'
train_df = df[df[fold_name] == 0]
train_df.reset_index(drop=True, inplace=True)
val_df = df[df[fold_name] == 1]
val_df.reset_index(drop=True, inplace=True)

# define the augmentation dataloader
batch_size = 16
# all_transforms = Compose(get_train_transform())
# crypto_train = CryptoDataset_Train(train_df, batch_size = batch_size)
# train_gen = MultiThreadedAugmenter(crypto_train, all_transforms, num_processes=4,\
                                #    num_cached_per_queue=2,\
                                #    seeds=None, pin_memory=True)
# crypto_val = CryptoDataset_Val(val_df)
crypto_train = CryptoDataset(train_df)
train_loader = DataLoader(crypto_train, batch_size = 16, shuffle=True, num_workers =4)
crypto_val = CryptoDataset(val_df)
val_loader = DataLoader(crypto_val, batch_size = 16, shuffle = False, num_workers = 4)

# load the pretrained model
model_depth = 50
n_classes = 1039
model = resnet.generate_model(model_depth=model_depth, n_input_channels = 2, n_classes=1039)
pretrain_model_path = './model/r3d50_KM_200ep.pth'
# pretrain_model = load_pretrained_model_1_c(model, pretrain_model_path, 'resnet50', 1)
pretrain_model = load_pretrained_model_2_c(model, pretrain_model_path, 'resnet50', 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrain_model = pretrain_model.to(device)

# freeze layers of the net
freeze_start = 9
pretrain_model = freezelayers(pretrain_model, freeze_start)

# set up the training
weight = torch.tensor([2.15]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
# optimizer = optim.SGD(pretrain_model.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, pretrain_model.parameters()), lr = 0.0001, momentum = 0.9)
check_points_save = '/data/ccusr/xinyug/lung/crypto/classification/checkpoints/vanilla_aug'
# if not os.path.exists(check_points_save):
#     os.mkdir(check_points_save)

# finetune the pretrained model
print('start finetune ...')
train_epoch_num = 300
model_result_dir = 'cube_64_2_c'
save_obj = save_results([],[],[],[],[],[],model_result_dir,fold_name)

max_auc = 0
for i in range(train_epoch_num):
    # train and validate the model
    start = time.time()
    train_loss = train_noaug(model = pretrain_model, optimizer=optimizer,\
                             criterion = criterion, train_loader = train_loader, device = device) 
    
    val_gt, val_pred , val_prob, val_loss = val_noaug(model = pretrain_model, criterion = criterion,\
                                                      val_loader= val_loader, device = device)

    # generate the results
    spe, sen, auc = getscore(val_gt, val_pred, val_prob) 
    end = time.time()
    epoch_time = round((end-start),2)
    spe = round(spe, 2)
    sen = round(sen, 2)
    auc = round(auc, 2)
    print('epoch ' + str(i) + ': spe ' + str(spe) + ', sen ' + str(sen) + ', auc ' + str(auc) + \
        ', train loss ' + str(train_loss) + ', val loss ' + str(val_loss) + \
        ', time ' + str(epoch_time)) 

    # save results
    # save_obj.add_item(spe, sen, auc, round(train_loss,2), round(val_loss,2),round(epoch_time,2))
    # if max_auc < auc:
    #     max_auc = auc
    #     save_obj.save_checkpoint(pretrain_model,i+1)

save_obj.save_to_csv()