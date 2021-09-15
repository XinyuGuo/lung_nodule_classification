import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import Compose
from utils.crypto_dataloader import CryptoDataset_Dual_Path_Train, CryptoDataset_Dual_Path_Val, get_train_transform
from model.dual_resnet_cls import dual_resnet_cls
from model import resnet_encoder
from utils.model_utils import train_dual_paths,val_dual_paths,save_results,getscore,BinaryFocalLoss
import pandas as pd
import math
import time
import pdb

# prepare dataloader
print('prepare the augment loader ...')
# csv_file = './csv/crypto_largest_nodule_199_cubes_5fold_balance.csv'
csv_file = '../csv/train_4.csv'
df = pd.read_csv(csv_file)

fold = 3
fold_name = 'fold_' + str(fold)
print('fold ' + str(fold))
train_df = df[df['fold']!= fold]
train_df.reset_index(drop=True, inplace=True)
val_df = df[df['fold'] == fold]
val_df.reset_index(drop=True, inplace=True)

# fold_name = 'fold_5'
# print(fold_name)
# train_df = df[df[fold_name] == 0]
# train_df.reset_index(drop=True, inplace=True)
# val_df = df[df[fold_name] == 1]
# val_df.reset_index(drop=True, inplace=True)

# define the augmentation dataloader
batch_size = 16
patch_size = (80,80,80)
all_transforms = Compose(get_train_transform(patch_size))
crypto_train = CryptoDataset_Dual_Path_Train(train_df, batch_size = batch_size)
train_gen = MultiThreadedAugmenter(crypto_train, all_transforms, num_processes=4,\
                                   num_cached_per_queue=2,\
                                   seeds=None, pin_memory=True)
crypto_val = CryptoDataset_Dual_Path_Val(val_df)
val_loader = DataLoader(crypto_val, batch_size = 16, shuffle = False, num_workers = 4)

# buid the image classifier
print('build the image classifer ...')
# the 3d resnet feature extractor
model_depth = 101
n_classes = 1039
# n_classes = 700
encoder_nodule = resnet_encoder.generate_model(model_depth=model_depth, n_input_channels = 1, n_classes=n_classes) # n_class = 1039
encoder_lung = resnet_encoder.generate_model(model_depth=model_depth, n_input_channels = 1, n_classes=n_classes)
pretrain_model_path = './model/r3d101_KM_200ep.pth'
# pretrain_model_path = './model/r3d101_K_200ep.pth'
fc_hidden = 128
fc_output = 1

cls_model = dual_resnet_cls(encoder_nodule,encoder_lung,pretrain_model_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cls_model = cls_model.to(device)


# train classifer
print('train the fc classifier ... ')
# set up the training
weight = torch.tensor([2.25]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight = weight)
# criterion = BinaryFocalLoss(alpha=[0.2,0.8])
optimizer = optim.SGD(filter(lambda p: p.requires_grad, cls_model.parameters()), lr = 0.001, momentum = 0.9, weight_decay=0.01)

# finetune the pretrained model
print('start finetune ...')
train_epoch_num = 300
batch_num = math.ceil(len(train_df)/batch_size)

max_auc = 0
save_obj = save_results([],[],[],[],[],[],'resnet101_1125_',fold_name)
for i in range(train_epoch_num):
    start = time.time()
    train_loss = train_dual_paths(model = cls_model, optimizer=optimizer,\
                                 criterion = criterion, train_gen = train_gen, batch_num = batch_num, device = device) 
    # pdb.set_trace()
    val_gt, val_pred , val_prob, val_loss = val_dual_paths(model = cls_model, criterion = criterion,\
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
        save_obj.save_checkpoint(cls_model,i+1)
save_obj.save_to_csv()
