import os
import argparse
from argparse import Namespace
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import esm
from esm.data import *
from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
from esm.model.esm2_supervised import ESM2
from esm.model.esm2_only_secondarystructure import ESM2 as ESM2_SS
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


import numpy as np
import pandas as pd
import random
import math
import scipy.stats as stats
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import (confusion_matrix, roc_auc_score, auc,
                             precision_recall_fscore_support,
                             precision_recall_curve, classification_report,
                             roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             accuracy_score)
from sklearn import preprocessing
from copy import deepcopy
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', type=str, default='0', help="Training Devices")
parser.add_argument('--local_rank', type=int, default=-1, help="DDP parameter, do not modify")
parser.add_argument('--seed', type = int, default = 1337)

parser.add_argument('--prefix', type=str, default = 'IRES_UTRLM')

parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--nodes', type = int, default = 40)
parser.add_argument('--dropout3', type = float, default = 0.5)
parser.add_argument('--lr', type = float, default = 1e-4)
parser.add_argument('--folds', type = int, default = 10)
parser.add_argument('--random_init', action = 'store_true')

parser.add_argument('--modelfile', type = str, default = 'UTRLM_pretrained_model.pkl')

parser.add_argument('--avg_emb', action = 'store_true') ## if --finetune: False
parser.add_argument('--bos_emb', action = 'store_true') ## if --finetune: False

parser.add_argument("--epoch_without_improvement", type=int, default=10, help="Early Stopping")
parser.add_argument("--pos_label_weight", type=float, default=-1, help="If is -1, calculate it based on the train labels")
parser.add_argument("--cls_loss_weight", type=float, default=20, help="If is -1, calculate it based on the train labels")
parser.add_argument("--mlm_loss_weight", type=float, default=1, help="If is -1, calculate it based on the train labels")
parser.add_argument("--finetune_esm", action='store_true', help='Flag to enable one-hot encoding or nn.Embedding()')
parser.add_argument("--finetune_sixth_layer_esm", action='store_true', help='Flag to enable one-hot encoding or nn.Embedding()')
parser.add_argument("--truncate", action='store_true', help='Flag to enable one-hot encoding or nn.Embedding()')
parser.add_argument("--truncate_num", type=int, default=50, help="Training batch size")
parser.add_argument("--mask_prob", type = float, default = 0.15, help = "0.15 or 0")
parser.add_argument("--batch_toks", type=int, default=2048, help="Training batch size")

args = parser.parse_args()
print(args)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

        
global layers, heads, embed_dim, batch_toks, inp_len, device_ids, device, train_obj_col, epoch, mask_toks_id, idx_to_tok
layers = 6
heads = 16
embed_dim = 128
batch_toks = args.batch_toks
folds = 10
repr_layers = [layers]
include = ["mean"]
return_contacts = False
return_representation = False

tok_to_idx = {'-': 0, '&': 1, '?': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '!': 7, '*': 8, '|': 9}
idx_to_tok = {idx: tok for tok, idx in tok_to_idx.items()}
print(tok_to_idx)
mask_toks_id = 8
# prefix = f'MJ3_seed{seed}_{args.prefix}'
    
device_ids = list(map(int, args.device_ids.split(',')))
dist.init_process_group(backend='nccl')
device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
torch.cuda.set_device(device)

# include = ["mean"]
# truncate = True

filename = f'{args.prefix}_{args.folds}folds_AvgEmb{args.avg_emb}_BosEmb{args.bos_emb}_epoch{args.epochs}_nodes{args.nodes}_dropout3{args.dropout3}_finetuneESM{args.finetune_esm}_finetuneLastLayerESM{args.finetune_sixth_layer_esm}_lr{args.lr}'
print(filename)
    
#### Model
class CNN_linear(nn.Module):
    def __init__(self):
        
        super(CNN_linear, self).__init__()
        
        self.embedding_size = embed_dim
        self.nodes = args.nodes
        self.dropout3 = args.dropout3
        
        if 'SISS' in args.modelfile:
            self.esm2 = ESM2_SISS(num_layers = layers,
                                     embed_dim = embed_dim,
                                     attention_heads = heads,
                                     alphabet = alphabet)
        elif 'SS' in args.modelfile:
            self.esm2 = ESM2_SS(num_layers = layers,
                                     embed_dim = embed_dim,
                                     attention_heads = heads,
                                     alphabet = alphabet)
        else:
            self.esm2 = ESM2(num_layers = layers,
                                     embed_dim = embed_dim,
                                     attention_heads = heads,
                                     alphabet = alphabet)
        
        self.dropout3 = nn.Dropout(self.dropout3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features = embed_dim, out_features = self.nodes)
        self.output = nn.Linear(in_features = self.nodes, out_features = 2)
            
    def forward(self, tokens, need_head_weights=True, return_contacts=True, return_representation = True):
        
        x = self.esm2(tokens, [layers], need_head_weights, return_contacts, return_representation)
        if args.avg_emb:
            x_o = x["representations"][layers][:, 1 : inp_len+1].mean(1)
            x_o = x_o.unsqueeze(2)
        elif args.bos_emb:
            x_o = x["representations"][layers][:, 0]
            x_o = x_o.unsqueeze(2)

        x_o = self.flatten(x_o)
        o_linear = self.fc(x_o)
        o_relu = self.relu(o_linear)
        o_dropout = self.dropout3(o_relu)
        o = self.output(o_dropout)
        
        return o, x['logits']
    
##### Running Step
def train_step(data, batches_loader, label_weight = None):

    model.train()
    y_prob_list, y_pred_list, y_true_list, loss_list, sample_list = [], [], [], [], []
    mlm_loss_seq_list, loss_seq_list = [], []
    sequence_list, predicted_sequence_list, label_list, y_prob_list, y_pred_list = [], [], [], [], []
    
    
    for i, batch in tqdm(enumerate(train_batches_loader)):
        batch = np.array(torch.LongTensor(batch))
        e_data = data.iloc[batch]
        
        dataset = FastaBatchedDataset(e_data.IRES_class_600, e_data.Sequence, mask_prob = args.mask_prob)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                collate_fn=alphabet.get_batch_converter(), 
                                                batch_size=len(batch), 
                                                shuffle = False)
   
        for (labels, sequences, masked_strs, toks, masked_toks, _) in dataloader:
            if args.truncate:
                toks = toks[:, :args.truncate_num]
                masked_toks = masked_toks[:, :args.truncate_num]
    
            toks = toks.to(device)
            masked_toks = masked_toks.to(device)
            
            one_hot_labels = torch.zeros((len(labels), 2))
            labels = torch.LongTensor(labels)
            one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
            labels = one_hot_labels.to(device)

            if label_weight is None: 
                label_weight = calculate_label_weight(labels).to(device)
       
            
            logit, esm_logit = model(masked_toks, return_representation = True, return_contacts=True)
            y_prob, y_pred, predicted_toks, predicted_sequence = cal_from_logit(logit, esm_logit)
            
            #### CLS LOSS
            loss_seq = args.cls_loss_weight * nn.CrossEntropyLoss(label_weight)(logit, labels)
            #### MLM Loss
            if args.mask_prob:
                toks.masked_fill_((masked_toks != mask_toks_id), -1)
                mlm_loss_seq = args.mlm_loss_weight * F.cross_entropy(esm_logit.transpose(1, 2), toks, ignore_index = -1, reduction = 'mean')
            else:
                mlm_loss_seq = torch.tensor(0).to(device)

            #### Loss
            loss = mlm_loss_seq + loss_seq 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.cpu().detach())
            mlm_loss_seq_list.append(mlm_loss_seq.cpu().detach())
            loss_seq_list.append(loss_seq.cpu().detach())

            y_prob_list.extend(y_prob.cpu().detach().tolist()) 
            y_pred_list.extend(y_pred.cpu().detach().tolist())
            y_true_list.extend(labels.cpu().detach()[:, 1].tolist()) 
            sample_list.extend(zip(sequences, labels.cpu().detach().tolist(), y_prob.cpu().detach().tolist(), y_pred.cpu().detach().tolist(), predicted_sequence))


    loss_epoch = float(torch.Tensor(loss_list).mean())
    mlm_loss_seq_list = float(torch.Tensor(mlm_loss_seq_list).mean())
    loss_seq_list = float(torch.Tensor(loss_seq_list).mean())
    print(f'Train: Epoch-{epoch}/{args.epochs} | Loss = {loss_epoch:.4f} | Loss_MLM = {mlm_loss_seq:.2f}|Loss_CLS = {loss_seq:.2f}')

    metrics = evaluate_metrics(loss_epoch, y_prob_list, y_pred_list, y_true_list)
    return metrics, (loss_epoch,mlm_loss_seq_list,loss_seq_list), sample_list


def predict_step(dataloader, model):
    model.eval()
    y_prob_list, y_pred_list, y_true_list, loss_list, sample_list = [], [], [], [], []
    mlm_loss_seq_list, loss_seq_list = [], []
    sequence_list, predicted_sequence_list, label_list, y_prob_list, y_pred_list = [], [], [], [], []
    with torch.no_grad():
        for (label, sequence, masked_sequence, sequence_token, masked_sequence_token, _) in tqdm(dataloader):
            
            if args.truncate:
                sequence_token = sequence_token[:, :args.truncate_num]
                masked_sequence_token = masked_sequence_token[:, :args.truncate_num]
            
            sequence_token = sequence_token.to(device)
            masked_sequence_token = masked_sequence_token.to(device)
            label = torch.LongTensor(label).to(device)

            logit, esm_logit = model.forward(sequence_token)
            y_prob, y_pred, predicted_toks, predicted_sequence = cal_from_logit(logit, esm_logit)
            #### CLS LOSS
            loss_seq = args.cls_loss_weight * nn.CrossEntropyLoss()(logit, label)  
            
            ### SAVE
            loss_seq_list.append(loss_seq.cpu().detach())

            y_prob_list.extend(y_prob.cpu().detach().tolist())
            y_pred_list.extend(y_pred.cpu().detach().tolist())
            y_true_list.extend(label.cpu().detach().tolist())
    
            sample_list.extend(zip(sequence, 
                                   label.cpu().detach().tolist(), y_prob.cpu().detach().tolist(), y_pred.cpu().detach().tolist(), 
                                   predicted_sequence))

        loss_seq_list = float(torch.Tensor(loss_seq_list).mean())
        
        print(f'Test: Epoch-{epoch}/{args.epochs} | Loss_CLS = {loss_seq_list:.4f}')

        metrics = evaluate_metrics(loss_seq_list, y_prob_list, y_pred_list, y_true_list)
    return metrics, loss_seq_list, pd.DataFrame(sample_list, 
columns = ['sequence', 'label', 'y_prob', 'y_pred', 'mutated_sequence'])

#### Performance Evaluation
def evaluate_metrics(loss, probs_list, preds_list, label_list):
    
    auc = roc_auc_score(label_list, probs_list)
    aupr = average_precision_score(label_list, probs_list)
    
    precision = precision_score(label_list, preds_list, zero_division=0)
    recall = recall_score(label_list, preds_list, zero_division=0)
    f1 = f1_score(label_list, preds_list, zero_division=0)
    accuracy = accuracy_score(label_list, preds_list)
    tn, fp, fn, tp = confusion_matrix(label_list, preds_list).ravel()

    # Store in DataFrame
    metrics_df = pd.DataFrame({
        'AUC': [auc],
        'AUPR': [aupr],
        'precision': [precision],
        'recall': [recall],
        'F1': [f1],
        'accuracy': [accuracy],
        'TN': [tn],
        'FP': [fp],
        'FN': [fn],
        'TP': [tp],
        'loss': [loss]
    })

    return metrics_df

def calculate_label_weight(labels):
    num_samples = len(labels)
    class_counts = torch.bincount(labels.long())
    
    # 处理全为0或全为1的情况
    if class_counts[0] == num_samples or class_counts[1] == num_samples:
        return torch.tensor([1.0, 1.0])
    
    class_weights = num_samples / (2 * class_counts.float())
    return class_weights

def cal_from_logit(logit, esm_logit):
    y_prob = torch.softmax(logit, dim = 1)
    y_pred = torch.argmax(y_prob, dim = 1)
    
    predicted_toks = esm_logit.argmax(dim=-1)
    predicted_tokens = [[idx_to_tok[idx.tolist()] for idx in seq] for seq in predicted_toks]
    predicted_sequence = [''.join(seq) for seq in predicted_tokens]
    
    return y_prob[:,1], y_pred, predicted_toks, predicted_sequence

#### Dataset

def generate_dataset_dataloader(e_data):
    dataset = FastaBatchedDataset(e_data.IRES_class_600, e_data.Sequence, mask_prob = args.mask_prob)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            collate_fn=alphabet.get_batch_converter(), 
                                            batch_sampler=batches, 
                                            shuffle = False)
    print(f"{len(dataset)} sequences")
    return dataset, dataloader


def generate_trainbatch_loader(e_data):
    dataset = FastaBatchedDataset(e_data.IRES_class_600, e_data.Sequence, mask_prob = args.mask_prob)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    batches_sampler = DistributedSampler(batches, shuffle = True)
    batches_loader = torch.utils.data.DataLoader(batches, 
                                                 batch_size = 1,
                                                 num_workers = 1,
                                                 sampler = batches_sampler)
    print(f"{len(dataset)} sequences")
    print(f'{len(batches)} batches')
    #print(f' Batches: {batches[0]}')
    return dataset, batches, batches_sampler, batches_loader
#######

alphabet = Alphabet(mask_prob = 0.0, standard_toks = 'AGCT')
print(alphabet.tok_to_idx)
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

print('====Load Data====')
data = pd.read_csv('../Data/v2_dataset_with_unified_stratified_shuffle_train_test_split.csv', index_col = 0)

metrics_df = pd.DataFrame()
train_loss_list, test_loss_list = [], []
for fold in range(args.folds):
    
    #### Read Data
    data_fold = data[data.fold == fold].sample(frac = 1)#.iloc[:1000]
    train_data = data_fold[data_fold.type == 'train'].reset_index(drop = True)#.iloc[:100]
    test_data = data_fold[data_fold.type == 'test']#.iloc[:100]

    train_dataset, train_batches, train_batches_sampler, train_batches_loader = generate_trainbatch_loader(train_data)
    test_dataset, test_dataloader = generate_dataset_dataloader(test_data)
    print(f'----Fold {fold}: ', len(train_data), len(test_data))
    
    ##### Label Weight
    train_labels = train_data['IRES_class_600'].values
    if args.pos_label_weight == -1:
        label_weight = calculate_label_weight(torch.Tensor(train_labels)).to(device)
    else:
        label_weight = torch.tensor([1.0, args.pos_label_weight]).to(device)
    
    ##### Load Model
    model = CNN_linear().to(device)
    storage_id = int(device_ids[args.local_rank])
    if not args.random_init:
        print(f'********Device IDs = {device_ids}, cuda:{device_ids[args.local_rank]}')
        model.esm2.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(f'../models/{args.modelfile}', map_location=lambda storage, loc : storage.cuda(storage_id)).items()}, strict = False)
        
    ##### FineTune?
    if not args.finetune_esm:
        for name, value in model.named_parameters():
            if args.finetune_sixth_layer_esm:
                if 'esm2' in name and 'esm2.layers.5' not in name:
                    value.requires_grad = False
            else:
                if 'esm2' in name:
                    value.requires_grad = False
    for name, value in model.named_parameters():                
        print(name, value.requires_grad)
            
    ##### Optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr = args.lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
        
    
    best_aupr = -1
    best_model_state = None
    epoch_without_improvement = 0
    train_loss_ep, test_loss_ep = [], []
    best_epoch_list = []
    for epoch in trange(1, args.epochs+1):
        train_batches_sampler.set_epoch(epoch)
        
        train_metrics, train_loss, train_result = train_step(train_data, train_batches_loader, label_weight=label_weight)
        train_loss_ep.extend(train_loss)

        test_metrics, test_loss, test_result = predict_step(test_dataloader, model)
        test_loss_ep.append(test_loss)

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: lr %.4f -> %.4f" % (epoch, before_lr, after_lr))
        
        print(f'Epoch {epoch}/{args.epochs} Train: ', end = '')
        for c in train_metrics.columns: print(f'{c}={train_metrics.loc[0, c]:.4f}', end = '|')
        print(f'\nEpoch {epoch}/{args.epochs} Test : ', end = '')
        for c in test_metrics.columns: print(f'{c}={test_metrics.loc[0, c]:.4f}', end = '|')
        print()
            
        if epoch == 1: model_best = deepcopy(model)
            
        if args.local_rank == 0:
            if test_metrics['AUPR'].values[0] > best_aupr:
                best_aupr = test_metrics['AUPR'].values[0]
                best_test_metrics = test_metrics
                best_train_metrics = train_metrics
                
                ep_best = epoch
                model_best = deepcopy(model)
                best_model_state = model.state_dict()
                torch.save(best_model_state, f"../IRES-UTRLM/models/{args.prefix}_best_model_fold{fold}.pt")
                epoch_without_improvement = 0
            else:
                epoch_without_improvement += 1

            if epoch_without_improvement >= args.epoch_without_improvement:
                break
                
   
                    
    ##### results
    if args.local_rank == 0:
        print('=====Generate results=====')
        train_dataset, train_dataloader = generate_dataset_dataloader(train_data)
        train_metrics, train_loss, train_result = predict_step(train_dataloader, model_best)
        test_metrics, test_loss, test_result = predict_step(test_dataloader, model_best)

        print('=====Loss=====')
        train_loss_list.append(train_loss_ep)
        test_loss_list.append(test_loss_ep)
        best_epoch_list.append(ep_best)

        print('====Save Metrics====')
        test_metrics.rename(columns=lambda x: "test_" + x, inplace=True)
        train_metrics.rename(columns=lambda x: "train_" + x, inplace=True)
        combined_metrics = pd.concat([test_metrics, train_metrics], axis=1)
        metrics_df = pd.concat([metrics_df, combined_metrics], axis=0)
        metrics_df.loc['mean'] = metrics_df.mean(axis = 0)
        metrics_df.loc['std'] = metrics_df.std(axis = 0)
        metrics_df.to_csv(f"../IRES-UTRLM/results/{filename}_metrics.csv")

        print('====Save y_pred====')
        train_result.to_csv(f'../IRES-UTRLM/y_pred/{filename}_train_fold{fold}.csv', index = False)
        test_result.to_csv(f'../IRES-UTRLM/y_pred/{filename}_test_fold{fold}.csv', index = False)

        ##### Figures
        fig, axes = plt.subplots(nrows = 2, ncols = 10, figsize = (40, 15))
        for i in range(fold):
            axes[0, i].plot(range(len(train_loss_list[i])), train_loss_list[i], label = f'Loss: Train_{i}')
            axes[1, i].plot(range(len(test_loss_list[i])), test_loss_list[i], label = f'Loss: Test_{i}')
            axes[0, i].legend(fontsize = 7)
            axes[1, i].legend(fontsize = 7)
        plt.savefig(f'../IRES-UTRLM/figures/{filename}.tif')