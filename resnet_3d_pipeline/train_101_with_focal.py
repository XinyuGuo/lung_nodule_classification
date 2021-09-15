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
from utils.model_utils import load_pretrained_model_1_c, freezelayers, train, val, getscore, save_results, BinaryFocalLoss, freezelayers_wo_bn
from utils.crypto_dataloader import CryptoDataset_Train, CryptoDataset_Val, get_train_transform

# prepare the data entry
print('prepare the augment loader ...')
csv_file = './csv/crypto_largest_nodule_199_cubes_5fold_shuffle.csv'
df = pd.read_csv(csv_file)
fold_name = 'fold_1'
print(fold_name)
train_df = df[df[fold_name] == 0]
train_df.reset_index(drop=True, inplace=True)
val_df = df[df[fold_name] == 1]
val_df.reset_index(drop=True, inplace=True)
# define the augmentation dataloader
batch_size = 16
patch_size = (80,80,80)
all_transforms = Compose(get_train_transform(patch_size))
crypto_train = CryptoDataset_Train(train_df, batch_size = batch_size)
train_gen = MultiThreadedAugmenter(crypto_train, all_transforms, num_processes=4,\
                                   num_cached_per_queue=2,\
                                   seeds=None, pin_memory=True)
crypto_val = CryptoDataset_Val(val_df)
val_loader = DataLoader(crypto_val, batch_size = 16, shuffle = False, num_workers = 4)

# load the pretrained model
model_depth = 101
n_classes = 1039
model = resnet.generate_model(model_depth=model_depth, n_input_channels = 1, n_classes=n_classes) # n_class = 1039
# pretrain_model_path = './model/r3d50_KM_200ep.pth'
# pretrain_model_path = './model/r3d18_K_200ep.pth'
# pretrain_model_path = './model/r3d34_K_200ep.pth'
pretrain_model_path = './model/r3d101_KM_200ep.pth'
# pretrain_model_path = './model/r3d152_KM_200ep.pth'
# pretrain_model_path = './model/r3d200_KM_200ep.pth'
pretrain_model = load_pretrained_model_1_c(model, pretrain_model_path, 'resnet101', 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrain_model = pretrain_model.to(device)

# freeze layers of the net
freeze_start = 9
# pretrain_model = freezelayers(pretrain_model, freeze_start)
pretrain_model = freezelayers_wo_bn(pretrain_model)
# set up the training  
criterion = BinaryFocalLoss(alpha=[0.2,0.8])
optimizer = optim.SGD(filter(lambda p: p.requires_grad, pretrain_model.parameters()), lr = 0.0001, momentum = 0.9)

# finetune the pretrained model
print('start finetune ...')
train_epoch_num = 150
batch_num = math.ceil(len(train_df)/batch_size)

max_auc = 0
save_obj = save_results([],[],[],[],[],[],'cube_80_1c_ResNet101_focal',fold_name)
for i in range(train_epoch_num):
    start = time.time()
    train_loss = train(model = pretrain_model, optimizer=optimizer,\
                       criterion = criterion, train_gen = train_gen, batch_num = batch_num, device = device) 
    
    val_gt, val_pred , val_prob, val_loss = val(model = pretrain_model, criterion = criterion,\
                                     val_loader= val_loader, device = device)
    
    spe, sen, auc = getscore(val_gt, val_pred, val_prob) 
    end = time.time()
    epoch_time = round((end-start),2)
    spe = round(spe, 2)
    sen = round(sen, 2)
    auc = round(auc, 2)
    print('epoch ' + str(i) + ': spe ' + str(spe) + ', sen ' + str(sen) + ', auc ' + str(auc) + \
          ', train loss ' + str(train_loss) + ', val loss ' + str(val_loss) + \
          ', time ' + str(epoch_time))
    save_obj.add_item(spe, sen, auc, round(train_loss,2), round(val_loss,2),round(epoch_time,2))
    if max_auc < auc:
        max_auc = auc
        save_obj.save_checkpoint(pretrain_model,i+1)
save_obj.save_to_csv()