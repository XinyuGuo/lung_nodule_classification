import torch
from torch import nn, sigmoid
import torch.nn.functional as F
# from torch import sigmoid
from sklearn.metrics import recall_score, roc_curve, auc, roc_auc_score
from collections import OrderedDict
import numpy as np 
from tqdm import tqdm
import pandas as pd
import os 
import pdb

def train_noaug(**kwargs):
    '''
    train model
    '''
    model = kwargs['model']
    optimizer = kwargs['optimizer']
    train_loader = kwargs['train_loader']
    criterion = kwargs['criterion']
    device = kwargs['device']
    running_loss = 0
    model.train()
    for train_batch in train_loader:
        # print(train_batch['pid'])
        # train_data = torch.tensor(normalize_img_arr(train_batch['data'])).to(device)
        train_data = train_batch['data'].to(device)
        train_label = train_batch['label'].to(device)
        train_label = torch.unsqueeze(train_label,1)
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    # pdb.set_trace()
    loss_mean = round(running_loss/len(train_loader),2)

    return loss_mean

def train_dual_paths(**kwargs):
    '''
    train model 
    ''' 
    def normalize_img_arr(img_arr):
        '''
        normalize 
        '''
        MIN_BOUND = -1000   
        MAX_BOUND = 200
        img_arr = (img_arr-MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        img_arr[img_arr>1]=1
        img_arr[img_arr<0]=0
        return img_arr

    model = kwargs['model']
    optimizer = kwargs['optimizer']
    train_gen = kwargs['train_gen']
    criterion = kwargs['criterion']
    batch_num = kwargs['batch_num']
    device = kwargs['device']
    running_loss = 0
    model.train()
    for i in range(batch_num):
        train_batch = train_gen.next()
        train_nodule = torch.tensor(normalize_img_arr(train_batch['data'])).to(device) 
        train_lung = torch.tensor(normalize_img_arr(train_batch['dual_lung'])).to(device) 
        train_label = torch.tensor(train_batch['label']).to(device)
        optimizer.zero_grad()
        output = model(train_nodule,train_lung)
        output = torch.squeeze(output)
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    loss_mean = round(running_loss/batch_num,2)
    return loss_mean

def val_dual_paths(**kwargs):
    '''
    model validation
    '''
    model = kwargs['model']
    criterion = kwargs['criterion']
    val_loader = kwargs['val_loader']
    device= kwargs['device']

    model.eval()
    gt = []
    pred = []
    prob = []
    with torch.no_grad():
        run_val_loss = 0
        for val_batch in val_loader:               
            val_data = val_batch['data'].to(device) 
            val_data = torch.unsqueeze(val_data,1)
            val_lung = val_batch['dual_lung'].to(device)
            val_lung = torch.unsqueeze(val_lung,1)
            val_label = val_batch['label'].to(device)
            output_val = model(val_data,val_lung)
            output_val = torch.squeeze(output_val,1)
            # gather the prediction and the ground truth
            pred_labels = sigmoid(output_val)
            pred_probs = pred_labels.clone()
            pred_labels[pred_labels > 0.5] = 1
            pred_labels[pred_labels <=0.5] = 0
            pred_labels = pred_labels.cpu().numpy().astype(np.uint8).tolist()
            pred_probs = pred_probs.cpu().numpy().astype(np.float32).tolist()
            gt_labels = val_label.cpu().numpy().astype(np.uint8).tolist()
            pred.extend(pred_labels)
            prob.extend(pred_probs)
            gt.extend(gt_labels)
            val_loss = criterion(output_val, val_label)
            run_val_loss+=val_loss.item()
        loss_mean = round(run_val_loss/len(val_loader),2)   
        return gt, pred, prob, loss_mean

def train(**kwargs):
    '''
    train model 
    ''' 
    def normalize_img_arr(img_arr):
        '''
        normalize 
        '''
        MIN_BOUND = -800   
        MAX_BOUND = 80
        img_arr = (img_arr-MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        img_arr[img_arr>1]=1
        img_arr[img_arr<0]=0
        return img_arr

    model = kwargs['model']
    optimizer = kwargs['optimizer']
    train_gen = kwargs['train_gen']
    criterion = kwargs['criterion']
    batch_num = kwargs['batch_num']
    device = kwargs['device']
    running_loss = 0
    model.train()
    for i in range(batch_num):
        train_batch = train_gen.next()
        train_data = torch.tensor(normalize_img_arr(train_batch['data'])).to(device) 
        train_label = torch.tensor(train_batch['label']).to(device)
        optimizer.zero_grad()
        output = model(train_data)
        output = torch.squeeze(output)
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    loss_mean = round(running_loss/batch_num,2)
    return loss_mean

def val_noaug(**kwargs):
    '''
    model validation
    '''
    model = kwargs['model']
    criterion = kwargs['criterion']
    val_loader = kwargs['val_loader']
    device= kwargs['device']

    model.eval()
    gt = []
    pred = []
    prob = []
    with torch.no_grad():
        run_val_loss = 0
        for val_batch in val_loader:      
            # print(val_batch['pid'])
            val_data = val_batch['data'].to(device) 
            val_label = val_batch['label'].to(device)
            gt_labels = val_label.cpu().numpy().astype(np.uint8).tolist()         
            gt.extend(gt_labels)
            val_label = torch.unsqueeze(val_label,1)
            output_val = model(val_data)
            val_loss = criterion(output_val, val_label)
            run_val_loss+=val_loss.item()
            # gather predicted labels
            output_val = torch.squeeze(output_val,1)
            pred_labels = torch.sigmoid(output_val)
            prob_labels = pred_labels.clone()
            pred_labels[pred_labels > 0.5] = 1
            pred_labels[pred_labels <=0.5] = 0
            pred_labels = pred_labels.cpu().numpy().astype(np.uint8).tolist()
            prob_labels = prob_labels.cpu().numpy().astype(np.float32).tolist()
            pred.extend(pred_labels)
            prob.extend(prob_labels)
            # print(pred)
            # pdb.set_trace()
        loss_mean = round(run_val_loss/len(val_loader),2)   
        return gt, pred, prob, loss_mean

def val(**kwargs):
    '''
    model validation
    '''
    model = kwargs['model']
    criterion = kwargs['criterion']
    val_loader = kwargs['val_loader']
    device= kwargs['device']

    model.eval()
    gt = []
    pred = []
    prob = []
    with torch.no_grad():
        run_val_loss = 0
        for val_batch in val_loader:               
            val_data = val_batch['data'].to(device) 
            val_data = torch.unsqueeze(val_data,1)
            val_label = val_batch['label'].to(device)
            output_val = model(val_data)
            output_val = torch.squeeze(output_val,1)
            # gather the prediction and the ground truth
            pred_labels = sigmoid(output_val)
            pred_probs = pred_labels.clone()
            pred_labels[pred_labels > 0.5] = 1
            pred_labels[pred_labels <=0.5] = 0
            pred_labels = pred_labels.cpu().numpy().astype(np.uint8).tolist()
            pred_probs = pred_probs.cpu().numpy().astype(np.float32).tolist()
            gt_labels = val_label.cpu().numpy().astype(np.uint8).tolist()
            pred.extend(pred_labels)
            prob.extend(pred_probs)
            gt.extend(gt_labels)
            val_loss = criterion(output_val, val_label)
            run_val_loss+=val_loss.item()
        loss_mean = round(run_val_loss/len(val_loader),2)   
        return gt, pred, prob, loss_mean

def change_channel_num(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key == 'conv1.weight':
            # value = torch.unsqueeze(value[:,1,:,:,:], 1)
            value = torch.unsqueeze(value[:,0,:,:,:], 1)
            # value = torch.unsqueeze(value[:,2,:,:,:], 1)
        new_state_dict[key] = value
    return new_state_dict

def load_pretrained_model_1_c(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')
        new_state_dict = change_channel_num(pretrain['state_dict'])
        model.load_state_dict(new_state_dict)
        tmp_model = model
        if model_name == 'densenet':
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                             n_finetune_classes)
        else:
            # print(tmp_model.fc.in_features)
            # pdb.set_trace()
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                     n_finetune_classes)
    return model

def change_channel_num_2(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key == 'conv1.weight':
            # value = torch.unsqueeze(value[:,0:2,:,:,:], 1)
            value = value[:,0:2,:,:,:]
            # print(value.shape)
            # pdb.set_trace()
        new_state_dict[key] = value
    return new_state_dict

def load_pretrained_model_2_c(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')
        new_state_dict = change_channel_num_2(pretrain['state_dict'])
        model.load_state_dict(new_state_dict)
        tmp_model = model
        if model_name == 'densenet':
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                             n_finetune_classes)
        else:
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                     n_finetune_classes)
    return model

def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')
        
        model.load_state_dict(pretrain['state_dict'])
        tmp_model = model
        if model_name == 'densenet':
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                             n_finetune_classes)
        else:
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                     n_finetune_classes)

    return model

def freezelayers_wo_bn(model):
    '''
    free all layers except for batch norm layers
    '''
    for name, p in model.named_parameters():
        if 'bn' not in name and 'fc' not in name:
            p.requires_grad = False 
    return model

def freezelayers(model,layer_num):
    '''
    freeze layers of the model, from layer_num to the botom layer
    '''
    ct = 0
    for child in model.children():
        ct += 1
        if ct < layer_num:
            for param in child.parameters():
                param.requires_grad = False  
    return model

def getscore(gt, pred, prob):
    '''
    calculate specificity and sensitivity
    '''
    score = recall_score(gt, pred, average=None, pos_label = 1, labels=[0,1], zero_division=0)
    auc_score = roc_auc_score(gt, prob)
    spe =score[0]
    sen =score[1]
    return spe, sen, auc_score

class save_results():
    '''
    save model sen, spe, auc, train_loss, val_loss and training epoch after one epoch
    '''
    def __init__(self, spes, sens, aucs, tls, vls, times, save_root, fold_name):
        self.spes = spes
        self.sens = sens 
        self.aucs = aucs
        self.tls = tls
        self.vls = vls
        self.times = times 
        self.file_name = fold_name + '.csv'
        self.save_path = os.path.join('/data/ccusr/xinyug/lung/cura_crypto/resnet_3d_pipeline/results',save_root)
        # dir containing model results (sen, spe, auc ....)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # dir containing model checkpoint
        self.checkpoint_dir = os.path.join(self.save_path, fold_name)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        
    def add_item(self, spe, sen, auc, tl, vl, time):
        self.spes.append(spe)
        self.sens.append(sen)
        self.aucs.append(auc)
        self.tls.append(tl)
        self.vls.append(vl)
        self.times.append(time)

    def save_to_csv(self):
        df = pd.DataFrame({'sen': self.sens,'spe':self.spes,'auc':self.aucs,\
                          'train_loss': self.tls, 'val_loss': self.vls, 'time': self.times})

        file_path = os.path.join(self.save_path, self.file_name)
        df.to_csv(file_path, index = False)

    def save_checkpoint(self, model, epoch_num):
        '''
        save model
        '''
        model_dict = {
            'state_dict':model.state_dict()
        }
        filename = 'checkpoint_'+str(epoch_num) + '_'+'.pth.tar'
        modelpath = os.path.join(self.checkpoint_dir,filename)
        torch.save(model_dict,modelpath)

def load_checkpoint(model, modelpath):
    '''
    load the model checkpoint
    '''
    model_params = torch.load(modelpath)
  
    # load parameters to the model.
    model.load_state_dict(model_params['state_dict'])
    model.eval()
    return model

class BinaryFocalLoss(nn.Module):
    """
    GITHUB: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/tree/master/FocalLoss
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=[1.0, 1.0], gamma=2, ignore_index=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert self.alpha.shape[0] == 2, \
                'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * \
                   torch.log(torch.sub(1.0, prob)) * neg_mask

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss
