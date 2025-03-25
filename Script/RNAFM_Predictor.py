# Import necessary libraries
import argparse
import numpy as np
import os
import pandas as pd
import pathlib
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm, trange

# Set global variables
seed = 19961231
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

global idx_to_tok, prefix, epochs, layers, heads, fc_node, dropout_prob, embed_dim, batch_toks, device, repr_layers, evaluation, include, truncate, return_contacts, return_representation, mask_toks_id, finetune

# prefix = f'ESM2_{args.prefix}'
layers = 6
heads = 16
embed_dim = 128
batch_toks = 1024
fc_node = 40
dropout_prob = 0.5
folds = 10
repr_layers = [-1]
include = ["mean"]
truncate = True
finetune = False
return_contacts = False
return_representation = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import sys
sys.path.append('/oak/stanford/scg/lab_congle/yanyichu/yanyichu/RNA-FM')
import fm
from fm.data import FastaBatchedDataset, Alphabet

_, alphabet = fm.pretrained.rna_fm_t12()
tok_to_idx = alphabet.tok_to_idx
idx_to_tok = {idx: tok for tok, idx in tok_to_idx.items()}
mask_toks_id = tok_to_idx['<mask>']
print(tok_to_idx)

class RNAFM_linear(nn.Module):
    def __init__(self):
        
        super(RNAFM_linear, self).__init__()
        
        self.embedding_size = 640 # embed_dim
        self.nodes = 40 #args.nodes
        self.dropout3 = 0.2 # args.dropout3
        
        self.rnafm, _ = fm.pretrained.rna_fm_t12()
        
        self.flatten = nn.Flatten()
        self.dropout3 = nn.Dropout(self.dropout3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features = self.embedding_size, out_features = self.nodes)
        self.output = nn.Linear(in_features = self.nodes, out_features = 2)
    
    def forward(self, tokens, return_repr = False):
        rnafm = self.rnafm(tokens, [12])
        x = rnafm["representations"]    #[12][:, ..., 1:-1, :]
        x_cls = x[12][:, 0]
        
        x_o = x_cls.unsqueeze(2)
    
        x_o = self.flatten(x_o)
        o_linear = self.fc(x_o)
        o_relu = self.relu(o_linear)
        o_dropout = self.dropout3(o_relu)
        o = self.output(o_dropout)
        
        y_prob = torch.softmax(o, dim = 1)
        y_pred = torch.argmax(y_prob, dim = 1)
        
#         return y_prob[:,1], y_pred, x_cls
        if return_repr:
            return y_prob[:, 1], y_pred, x_cls
        else:
            return y_prob[:, 1], y_pred
        
    def predict(self, tokens, transform_type = ''):
        rnafm = self.rnafm(tokens, [12])
        x = rnafm["representations"]    #[12][:, ..., 1:-1, :]

        x_cls = x[12][:, 0]
        x_o = x_cls.unsqueeze(2)
    
        x_o = self.flatten(x_o)
        o_linear = self.fc(x_o)
        o_relu = self.relu(o_linear)
        o_dropout = self.dropout3(o_relu)
        o = self.output(o_dropout)
        
        y_prob = torch.softmax(o, dim = 1)
        y_pred = torch.argmax(y_prob, dim = 1)
        
        if transform_type:
            y_prob_transformed = prob_transform(y_prob[:,1])
            return y_prob[:,1], y_pred, rnafm['logits'], y_prob_transformed
        else:
            return y_prob[:,1], y_pred, rnafm['logits'], o[:,1]

def generate_dataset_dataloader(label, sequence):
    sequence = [s.replace('T', 'U') for s in sequence]
    dataset = FastaBatchedDataset(label, sequence, mask_prob = 0)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            collate_fn=alphabet.get_batch_converter(), 
                                            batch_sampler=batches, 
                                            shuffle = False)
    #print(f"{len(dataset)} sequences")
    return dataset, dataloader

def predict_step(dataloader, model, fold, return_repr = False):
    model.eval()
    y_prob_list, y_pred_list, sample_list = [], [], []
    x_repr_list, x_attn_list = [], []
    with torch.no_grad():
        for (label, sequence, _, sequence_token, _, _) in dataloader:

            sequence_token = sequence_token[:, :1024]
            
            sequence_token = sequence_token.to(device)
#             y_prob, y_pred, x_repr = model.forward(sequence_token, return_repr)
            if return_repr:
                y_prob, y_pred, x_repr = model.forward(sequence_token, return_repr=True)
                x_repr_list.append(x_repr.cpu().detach())
            else:
                y_prob, y_pred = model.forward(sequence_token)
                
            y_prob_list.extend(y_prob.cpu().detach().tolist())
            y_pred_list.extend(y_pred.cpu().detach().tolist())
            sample_list.extend(zip(sequence, label, y_prob.cpu().detach().tolist(), y_pred.cpu().detach().tolist()))
            
    result = {
        'df': pd.DataFrame(sample_list, columns=['sequence', 'category', f'Prob_R{fold}', f'Pred_R{fold}'])
    }
    
    if return_repr:
        result['x_repr'] = torch.cat(x_repr_list, dim=0).numpy()
    return result
#     return pd.DataFrame(sample_list, columns = ['sequence', 'category', f'Prob_R{fold}', f'Pred_R{fold}']), x_repr_concat
