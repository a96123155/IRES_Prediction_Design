# Import necessary libraries
import numpy as np
import os
import pandas as pd
import pathlib
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from esm.data import *
from esm.model.esm2 import ESM2
from torch import nn
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


global idx_to_tok, prefix, epochs, layers, heads, fc_node, dropout_prob, embed_dim, batch_toks, device, repr_layers, evaluation, include, truncate, return_contacts, return_representation, mask_toks_id

# prefix = f'ESM2_{args.prefix}'
layers = 6
heads = 16
embed_dim = 128
batch_toks = 4096
fc_node = 40
dropout_prob = 0.5
folds = 10
repr_layers = [-1]
include = ["mean"]
truncate = True
return_contacts = False
return_representation = False

# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global tok_to_idx, idx_to_tok
alphabet = Alphabet(mask_prob = 0.15, standard_toks = 'AGCT')
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

# tok_to_idx = {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}
tok_to_idx = {'-': 0, '&': 1, '?': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '!': 7, '*': 8, '|': 9}
idx_to_tok = {idx: tok for tok, idx in tok_to_idx.items()}


class CNN_linear(nn.Module):
    def __init__(self):
        super(CNN_linear, self).__init__()
        
        self.esm2 = ESM2(num_layers = layers,
                         embed_dim = embed_dim,
                         attention_heads = heads,
                         alphabet = alphabet)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features = embed_dim, out_features = fc_node)
        self.output = nn.Linear(in_features = fc_node, out_features = 2)        
            
    def forward(self, tokens, return_repr = False, return_attn = False):
        
        x = self.esm2(tokens, [layers], need_head_weights=True, return_contacts=True, return_representation = True)
        x_cls = x["representations"][layers][:, 0]
        
        o = self.fc(x_cls)
        o = self.relu(o)
        o = self.dropout(o)
        o = self.output(o)
        
        y_prob = torch.softmax(o, dim = 1)
        y_pred = torch.argmax(y_prob, dim = 1)
        
#         return y_prob[:,1], y_pred, x_cls, x['attentions_symm'].sum(-1) #, x['logits'], o[:,1], 
        if return_repr and return_attn:
            return y_prob[:, 1], y_pred, x_cls, x['attentions_symm'].sum(-1)
        elif return_repr:
            return y_prob[:, 1], y_pred, x_cls
        elif return_attn:
            return y_prob[:, 1], y_pred, x['attentions_symm'].sum(-1)
        else:
            return y_prob[:, 1], y_pred
        
    def predict(self, tokens, transform_type = ''):
        
        x = self.esm2(tokens, [layers], need_head_weights=True, return_contacts=True, return_representation = True)
        x_cls = x["representations"][layers][:, 0]
        
        o = self.fc(x_cls)
        o = self.relu(o)
        o = self.dropout(o)
        o = self.output(o)
        
        y_prob = torch.softmax(o, dim = 1)
        y_pred = torch.argmax(y_prob, dim = 1)
        
        if transform_type:
            y_prob_transformed = prob_transform(y_prob[:,1])
            return y_prob[:,1], y_pred, x['logits'], y_prob_transformed#, x['attentions_symm'].sum(-1)
        else:
            return y_prob[:,1], y_pred, x['logits'], o[:,1]#, x['attentions_symm'].sum(-1)
    

def generate_dataset_dataloader(label, sequence):
    dataset = FastaBatchedDataset(label, sequence, mask_prob = 0)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            collate_fn=alphabet.get_batch_converter(), 
                                            batch_sampler=batches, 
                                            shuffle = False)
#     print(f"{len(dataset)} sequences")
    return dataset, dataloader


def predict_step(dataloader, model, fold, return_repr = False, return_attn = False):
    model.eval()
    y_prob_list, y_pred_list, sample_list = [], [], []
    x_repr_list, x_attn_list = [], []
    with torch.no_grad():
        for (label, sequence, _, sequence_token, _, _) in dataloader:
            sequence_token = sequence_token.to(device)

            if return_repr and return_attn:
                y_prob, y_pred, x_repr, x_attn = model.forward(sequence_token, return_repr=True, return_attn=True)
                x_repr_list.append(x_repr.cpu().detach())
                x_attn_list.append(x_attn.cpu().detach())
            elif return_repr:
                y_prob, y_pred, x_repr = model.forward(sequence_token, return_repr=True)
                x_repr_list.append(x_repr.cpu().detach())
            elif return_attn:
                y_prob, y_pred, x_attn = model.forward(sequence_token, return_attn=True)
                x_attn_list.append(x_attn.cpu().detach())
            else:
                y_prob, y_pred = model.forward(sequence_token)
                
#             y_prob, y_pred, x_repr, x_attn = model.forward(sequence_token, return_repr, return_attn)

            y_prob_list.extend(y_prob.cpu().detach().tolist())
            y_pred_list.extend(y_pred.cpu().detach().tolist())
            sample_list.extend(zip(sequence, label, y_prob.cpu().detach().tolist(), y_pred.cpu().detach().tolist()))
            
    result = {
        'df': pd.DataFrame(sample_list, columns=['sequence', 'category', f'Prob_U{fold}', f'Pred_U{fold}'])
    }
    
    if return_repr:
        result['x_repr'] = torch.cat(x_repr_list, dim=0).numpy()
    if return_attn:
        result['x_attn'] = torch.cat(x_attn_list, dim=0).numpy()
    
    return result

#     return pd.DataFrame(sample_list, columns = ['sequence', 'category', f'Prob_U{fold}', f'Pred_U{fold}']), x_repr_concat, x_attn_concat

# dataset, dataloader = generate_dataset_dataloader(label, sequence)
# utrlm_result = predict_step(dataloader, model)
