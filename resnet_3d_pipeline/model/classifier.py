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
    def __init__(self, hidden_num, output_num, input_num = 512):
        super().__init__()
        self.fc_hidden = nn.Linear(input_num, hidden_num)
        self.avepool1d = nn.AvgPool1d(4,stride=4)
        self.maxpool1d = nn.MaxPool1d(4,stride=4)
        self.fc_output = nn.Linear(hidden_num,output_num)

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.maxpool1d(x)
        x = torch.squeeze(x,1)
        hidden_feats =self.fc_hidden(x)
        output = self.fc_output(hidden_feats)
        return output 

class fc_head_2(nn.Module):
    '''
    a three-layer fc classifer 
    '''
    def __init__(self, output_num, input_num = 512):
        super().__init__()
        self.avepool1d = nn.AvgPool1d(4,stride=4)
        self.maxpool1d = nn.MaxPool1d(4,stride=4)
        self.fc_output = nn.Linear(input_num,output_num)

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.avepool1d(x)
        x = torch.squeeze(x,1)
        output = self.fc_output(x)
        return output 


class classifier(nn.Module):
    '''
    the resnet encoder plus the fc classifer
    '''
    def __init__(self, encoder, pretrain_encoder_path, hidden_num, output_num, freeze_layer=9):
        super().__init__()
        self.freeze_layer = freeze_layer
        self.encoder = encoder
        pretrain_encoder = load_pretrained_model_1_c(encoder, pretrain_encoder_path, 'resnet', 1)
        self.pretrain_encoder = self._freeze_encoder(pretrain_encoder)
        self.fc_head = fc_head(hidden_num, output_num)
    
    def forward(self,x):
        _, feats = self.pretrain_encoder(x)
        
        pred = self.fc_head(feats)
        return pred

    def _freeze_encoder(self, pretrain_encoder):
        cnt = 0
        for child in pretrain_encoder.children():
            cnt+=1
            if cnt < self.freeze_layer:
                for param in child.parameters():

                    param.requires_grad = False  
        return pretrain_encoder

class classifier_2(nn.Module):
    '''
    the resnet encoder plus the fc classifer
    '''
    def __init__(self, encoder, pretrain_encoder_path, output_num, freeze_layer=9):
        super().__init__()
        self.freeze_layer = freeze_layer
        self.encoder = encoder
        pretrain_encoder = load_pretrained_model_1_c(encoder, pretrain_encoder_path, 'resnet', 1)
        self.pretrain_encoder = self._freeze_encoder(pretrain_encoder)
        self.fc_head = fc_head_2(output_num)
    
    def forward(self,x):
        _, feats = self.pretrain_encoder(x)
        
        pred = self.fc_head(feats)
        return pred

    def _freeze_encoder(self, pretrain_encoder):
        cnt = 0
        for child in pretrain_encoder.children():
            cnt+=1
            if cnt < self.freeze_layer:
                for param in child.parameters():
                    print(param)
                    pdb.set_trace()
                    param.requires_grad = False  
        return pretrain_encoder      