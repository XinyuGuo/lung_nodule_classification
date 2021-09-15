import torch.nn as nn
import torch
import sys
sys.path.append('../utils')
from utils.model_utils import load_pretrained_model_1_c
import pdb
class fc_head(nn.Module):
    '''
    a three-layer fc classifer 
    '''
    def __init__(self):
        super().__init__()
        self.maxpool1d = nn.MaxPool1d(2,stride=2)
        self.fc_output = nn.Linear(2048,1)

    def forward(self, nodule_f, lung_f):
        nodule_f = torch.unsqueeze(nodule_f,1)
        lung_f = torch.unsqueeze(lung_f,1)
        nodule_f= self.maxpool1d(nodule_f)
        lung_f = self.maxpool1d(lung_f)
        fusion = torch.cat((nodule_f,lung_f),2)
        fusion = torch.squeeze(fusion,1)
        output = self.fc_output(fusion)
        return output 

class dual_resnet_cls(nn.Module):
    '''
    the resnet encoder plus the fc classifer
    '''
    def __init__(self, encoder_nodule, encoder_lung, pretrain_encoder_path):
        super().__init__()
        pretrain_encoder_nodule = load_pretrained_model_1_c(encoder_nodule, pretrain_encoder_path, 'resnet', 1)
        pretrain_encoder_lung = load_pretrained_model_1_c(encoder_lung,pretrain_encoder_path,'resnet', 1)
        self.pretrain_encoder_nodule = self.freezelayers(pretrain_encoder_nodule, 9)
        self.pretrain_encoder_lung = self.freezelayers(pretrain_encoder_lung, 9)
        self.fc_head = fc_head()
    
    def forward(self,nodule_t, lung_t):
        _,feats_nodule = self.pretrain_encoder_nodule(nodule_t)
        _,feats_lung = self.pretrain_encoder_lung(lung_t)
        pred = self.fc_head(feats_nodule,feats_lung)
        return pred

    def freezelayers(self, model,layer_num):
        '''
        freeze layers of the model, from layer_num to the botom layer
        '''
        ct = 0
        for child in model.children():
            if ct < layer_num:
                for param in child.parameters():
                    param.requires_grad = False  
            ct += 1
        return model

    def unfreezemodels(self,layer_num):
        '''
        open layers to fine tune the model
        '''
        ct = 0
        for child in self.pretrain_encoder_nodule.children():
            ct += 1
            if ct < layer_num:
                for param in child.parameters():
                    param.requires_grad = True  
        
        ct = 0 
        for child in self.pretrain_encoder_lung.children():
            ct += 1
            if ct < layer_num:
                for param in child.parameters():
                    param.requires_grad = True  


class dual_resnet_cls_eval(nn.Module):
    '''
    the resnet encoder plus the fc classifer
    '''
    def __init__(self, encoder_nodule, encoder_lung):
        super().__init__()
        self.pretrain_encoder_nodule = encoder_nodule
        self.pretrain_encoder_lung = encoder_lung
        self.fc_head = fc_head()
    
    def forward(self,nodule_t, lung_t):
        _,feats_nodule = self.pretrain_encoder_nodule(nodule_t)
        _,feats_lung = self.pretrain_encoder_lung(lung_t)
        pred = self.fc_head(feats_nodule,feats_lung)
        return pred

    def unfreezemodels(self,layer_num):
        '''
        open layers to fine tune the model
        '''
        ct = 0
        for child in self.pretrain_encoder_nodule.children():
            ct += 1
            if ct < layer_num:
                for param in child.parameters():
                    param.requires_grad = True  
        
        ct = 0 
        for child in self.pretrain_encoder_lung.children():
            ct += 1
            if ct < layer_num:
                for param in child.parameters():
                    param.requires_grad = True  