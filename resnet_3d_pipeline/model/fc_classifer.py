import torch.nn as nn
import sys
sys.path.append('../utils')
from utils.model_utils import load_pretrained_model_1_c

class fc_head(nn.Module):
    '''
    a three-layer fc classifer 
    '''
    def __init__(self, input_num = 2048, hidden_num, output_num):
        super().__init__()
        self.fc_hidden = nn.Linear(input_num, hidden_num)
        self.fc_output = nn.Linear(hidden_num,output_num)

    def forward(self, x):
        hidden_feats = self.fc_hidden(x)
        output = self.fc_output(hidden_feats)
        return output 

class classifier(nn.Module):
    '''
    the resnet encoder plus the fc classifer
    '''
    def __init__(self, encoder, pretrain_encoder_path, hidden_num, output_num):
        self.encoder = encoder
        pretrain_encoder = load_pretrained_model_1_c(encoder, pretrain_model_path, 'resnet', 1)
        self.pretrain_encoder = self.freeze_encoder(pretrain_encoder)
        self.fc_head = fc_head(hidden_num, output_num)
    
    def forward(self,x):
        
        feats = self.pretrain_encoder(x)
        pred = fc_head(feats)
        return feats

    def freeze_encoder(self, pretrain_encoder):
        print(pretrain_encoder)
        
        