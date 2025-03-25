import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import random
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from collections import Counter, OrderedDict
from copy import deepcopy
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from esm.data import *
from esm.model.esm2 import ESM2
from sklearn import preprocessing
from sklearn.metrics import (confusion_matrix, roc_auc_score, auc,
                             precision_recall_fscore_support,
                             precision_recall_curve, classification_report,
                             roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             accuracy_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from scipy.stats import spearmanr, pearsonr
from torch import nn
from torch.nn import Linear
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm, trange

import vMay7_UTRLM_Predictor as utrlm

matplotlib.rcParams.update({'font.size': 7})
def seed_everything(seed=42):
    """ "
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


global idx_to_tok, prefix, epochs, layers, heads, fc_node, dropout_prob, embed_dim, batch_toks, device, repr_layers, evaluation, include, truncate, return_contacts, return_representation, mask_toks_id, finetune

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
finetune = False
return_contacts = False
return_representation = False

# Set up environment variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global tok_to_idx, idx_to_tok, mask_toks_id
alphabet = Alphabet(mask_prob = 0.15, standard_toks = 'AGCT')
assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

tok_to_idx = {'-': 0, '&': 1, '?': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '!': 7, '*': 8, '|': 9}
idx_to_tok = {idx: tok for tok, idx in tok_to_idx.items()}
print(tok_to_idx)
mask_toks_id = 8

import argparse


def parse_dict(string):
    # 解析形如 "key1=value1,key2=value2" 的字符串为字典
    # 忽略空字符串，返回空字典
    if not string:
        return {}
    return dict(x.split("=") for x in string.split(","))

parser = argparse.ArgumentParser(description="Training configurations for SeqOnehotCNN")
parser.add_argument("--prefix", type=str, default='VCIP_gene_editing_AtoG', help="Prefix for saved models")
parser.add_argument("--wt_seq", type=str, default='CAGCCTCGGCCAGGAGGCGACCCGGGCGCCTGGGTGTGTGGCTGCTGTTGCGGGACGTCTTCGCGGGGCGGGAGGCTCGCGCCGCAGCCAGCGCC', help="Default is VCIP")
parser.add_argument("--nb_folds", type=int, default=10, help="How many models of cross-validation")
parser.add_argument("--model_filename", type=str, default='../models/IRES_UTRLM_best_model_fold0.pt', help="Prefix for saved models")

parser.add_argument("--start_nt_position", type=int, default=0, help="The first position is 0")
parser.add_argument("--end_nt_position", type=int, default=-1, help="The last position is length(sequence)-1")
parser.add_argument("--mut_by_prob", action='store_true', help='Flag to enable one-hot encoding or nn.Embedding()')
parser.add_argument("--transform_type", type=str, default='logit', choices=['', 'sigmoid', 'logit', 'power_law', 'tanh'], help="Type of probability transformation")

parser.add_argument("--mlm_tok_num", type=int, default=1, help="Number of masked tokens for each sequence per epoch")
parser.add_argument("--n_mut", type=int, default=3, help="Maximum number of mutations for each sequence")
parser.add_argument("--n_designs_ep", type=int, default=10, help="Number of mutations per epoch")
parser.add_argument("--n_sampling_designs_ep", type=int, default=5, help="Number of sampling mutations from n_designs_ep per epoch")
parser.add_argument("--n_mlm_recovery_sampling", type=int, default=1, help="There are AGCT could be recovery, Number of MLM recovery samplings")
parser.add_argument("--mutate2stronger", action='store_true', help="Flag indicating whether to mutate to stronger variant")
parser.add_argument('--mutation_type', choices=['mut_attn', 'mut_gene_editing', 'mut_random'], default = 'mut_random', 
                    help='Select from mut_attn, mut_gene_editing, mut_random')
parser.add_argument('--nt_replacements', default='A=G', type=parse_dict,
                    help='For mut_gene_editing, e.g., "A=G,C=T"')
args = parser.parse_args()

if not args.mut_by_prob and args.transform_type != '':
    print("--transform_type must be '' when --mut_by_prob is False")
    args.transform_type = ''
    
print(args)


def prob_transform(prob, **kwargs): # Logits
    """
    Transforms probability values based on the specified method.

    :param prob: torch.Tensor, the input probabilities to be transformed
    :param transform_type: str, the type of transformation to be applied
    :param kwargs: additional parameters for transformations
    :return: torch.Tensor, transformed probabilities
    """

    if args.transform_type == 'sigmoid':
        x0 = kwargs.get('x0', 0.5)
        k = kwargs.get('k', 10.0)
        prob_transformed = 1 / (1 + torch.exp(-k * (prob - x0)))
    
    elif args.transform_type == 'logit':
        # Adding a small value to avoid log(0) and log(1)
        prob_transformed = torch.log(prob + 1e-6) - torch.log(1 - prob + 1e-6)
    
    elif args.transform_type == 'power_law':
        gamma = kwargs.get('gamma', 2.0)
        prob_transformed = torch.pow(prob, gamma)
    
    elif args.transform_type == 'tanh':
        k = kwargs.get('k', 2.0)
        prob_transformed = torch.tanh(k * prob)
    
    return prob_transformed

def align_sequences(wt_seq, mt_seqs):
    # 这个函数假定两个字符串的长度相同。
    res = []
    for mt in mt_seqs:
        mt_align, mut_n = '', 0
        for s1, s2 in zip(wt_seq, mt):
            mt_align += ['-', s2][s1 != s2]
            mut_n += [0, 1][s1 != s2]
        res.append([mut_n, mt_align, wt_seq, mt])
    return pd.DataFrame(res, columns = ['Mut_Num', 'Mut_Align', 'WT', 'MT']).reset_index(drop = True)

def attention_based_replace(sequence, attention_weights, continuous_replace=False):
    if args.end_nt_position == -1: args.end_nt_position = len(sequence)
    if args.start_nt_position < 0 or args.end_nt_position > len(sequence) or args.start_nt_position > args.end_nt_position:
        print("Invalid start/end positions")
        args.start_nt_position, args.end_nt_position = 0, -1
    
    # 将序列切片成三部分：替换区域前、替换区域、替换区域后
    pre_segment = sequence[:args.start_nt_position]
    target_segment = list(sequence[args.start_nt_position:args.end_nt_position + 1])  # +1因为Python的切片是右开区间
    post_segment = sequence[args.end_nt_position + 1:]
    
    # 提取对应于目标片段的注意力权重
    attention_weights = attention_weights[0]
    target_attention_weights = attention_weights[args.start_nt_position:args.end_nt_position + 1]
    if not continuous_replace:
        # 根据注意力权重选择mlm_tok_num个位置
        # 注意：这里假设更高的权重意味着更大的替换概率
        indices = sorted(range(len(target_attention_weights)), key=lambda i: target_attention_weights[i], reverse=True)[:args.mlm_tok_num]
        for idx in indices:
            target_segment[idx] = '*'
    else:
        # 如果需要连续替换，则选择具有最高平均注意力权重的mlm_tok_num连续位置
        max_avg_weight, best_start_idx = -1, 0
        for i in range(len(target_segment) - args.mlm_tok_num + 1):
            avg_weight = sum(target_attention_weights[i:i+args.mlm_tok_num]) / args.mlm_tok_num
#             print(i, i+args.mlm_tok_num, target_attention_weights[i:i+args.mlm_tok_num].shape)
            
            if avg_weight > max_avg_weight:
                max_avg_weight, best_start_idx = avg_weight, i
        for idx in range(best_start_idx, best_start_idx + args.mlm_tok_num):
            target_segment[idx] = '*'
    
    # 合并并返回最终的序列
    return ''.join([pre_segment] + target_segment + [post_segment])

def gene_editing_replace(sequence, nt_replacements, start_nt_position=0, end_nt_position=-1, mlm_tok_num=1):
    if end_nt_position == -1 or end_nt_position > len(sequence):
        end_nt_position = len(sequence)
    if start_nt_position < 0 or start_nt_position > end_nt_position:
        print("Invalid start/end positions")
        return sequence
    
    # 将序列切片成三部分：替换区域前、替换区域、替换区域后
    pre_segment = sequence[:start_nt_position]
    target_segment = list(sequence[start_nt_position:end_nt_position])
    post_segment = sequence[end_nt_position:]
    
    # 确定所有要替换的核苷酸的位置
    indices = [i for i, nt in enumerate(target_segment) if nt in nt_replacements]

    # 如果替换数量少于找到的核苷酸，随机选取部分进行替换
    if mlm_tok_num < len(indices):
        indices = random.sample(indices, mlm_tok_num)
    
    # 进行替换
    for idx in indices:
        original_nt = target_segment[idx]
        target_segment[idx] = nt_replacements.get(original_nt, original_nt)  # 获取替换核苷酸，如果不存在则保留原核苷酸
    
    # 合并并返回最终的序列
    return ''.join([pre_segment] + target_segment + [post_segment])


def random_replace(tok, continuous_replace=False):
    sequence = deepcopy(tok[0])
    start_nt_position = args.start_nt_position
    end_nt_position = args.end_nt_position

    if end_nt_position == -1:
        end_nt_position = len(sequence)
    if start_nt_position < 0 or end_nt_position > len(sequence) or start_nt_position > end_nt_position:
        print("Invalid start/end positions")
        start_nt_position, end_nt_position = 0, len(sequence) - 1
    
    # 将序列切片成三部分：替换区域前、替换区域、替换区域后
    pre_segment = sequence[:start_nt_position]
    target_segment = sequence[start_nt_position:end_nt_position + 1]  # +1因为Python的切片是右开区间
    post_segment = sequence[end_nt_position + 1:]
    
    if not continuous_replace:
        # 随机替换目标片段的 mlm_tok_num 个位置
        indices = random.sample(range(len(target_segment)), args.mlm_tok_num)
        for idx in indices:
            target_segment[idx] = 8
    else:
        # 在目标片段连续替换 mlm_tok_num 个位置
        max_start_idx = len(target_segment) - args.mlm_tok_num  # 确保从 start_idx 开始的 mlm_tok_num 个元素不会超出目标片段的长度
        if max_start_idx < 1:  # 如果目标片段长度小于 mlm_tok_num，返回原始序列
            return sequence
        start_idx = random.randint(0, max_start_idx)
        for idx in range(start_idx, start_idx + args.mlm_tok_num):
            target_segment[idx] = 8
    
    # 合并并返回最终的序列
    return torch.cat([pre_segment, target_segment, post_segment])

def mlm_seq(seq):
    seq_token, masked_sequence_token = [7],[7]
    seq_token += [tok_to_idx[token] for token in seq]
    
    if args.mutation_type=='mut_attn':
        masked_seq = attention_based_replace(seq, attn, continuous_replace)
    elif args.mutation_type=='mut_random':
        masked_seq = random_replace(seq, continuous_replace)  # 随机替换n_mut个元素为'*'
#     masked_seq = random_replace(seq, args.n_mut) # 随机替换n_mut个元素为'*'
    elif args.mutation_type=='mut_gene_editing':
        masked_seq = gene_editing_replace(seq, args.nt_replacements)  # 随机替换n_mut个元素为'*'
    masked_seq_token += [tok_to_idx[token] for token in masked_seq]
    
    return seq, masked_seq, torch.LongTensor(seq_token), torch.LongTensor(masked_seq_token)


def batch_mlm_seq(seed_tok, attn = None, continuous_replace = False):

    batch_masked_seq_token_list = []

    for i in range(args.n_designs_ep):

        if args.mutation_type=='mut_attn':
            masked_tok = attention_based_replace(seq, attn, continuous_replace)
        elif args.mutation_type=='mut_random':
            masked_tok = random_replace(seed_tok, continuous_replace)  # 随机替换n_mut个元素为'*'
        elif args.mutation_type=='mut_gene_editing':
            masked_tok = gene_editing_replace(seq, args.nt_replacements)  # 随机替换n_mut个元素为'*'
        batch_masked_seq_token_list.append(masked_tok)
    return torch.stack(batch_masked_seq_token_list)

def recovered_mlm_tokens(masked_seqs, masked_toks, esm_logits, exclude_low_prob = False):
    # Only remain the AGCT logits
    esm_logits = esm_logits[:,:,3:7]
    # Get the predicted tokens using argmax
    predicted_toks = (esm_logits.argmax(dim=-1)+3).tolist()
    
    batch_size, seq_len, vocab_size = esm_logits.size()
    if exclude_low_prob: min_prob = 1 / vocab_size
    # Initialize an empty list to store the recovered sequences
    recovered_sequences, recovered_toks = [], []
    
    for i in range(batch_size):
        recovered_sequence_i, recovered_tok_i = [], []
        for j in range(seq_len):
            if masked_toks[i][j] == 8:
                print(i,j)
                ### Sample M recovery sequences using the logits
                recovery_probs = torch.softmax(esm_logits[i, j], dim=-1)
                recovery_probs[predicted_toks[i][j]-3] = 0  # Exclude the most probable token
                if exclude_low_prob: recovery_probs[recovery_probs < min_prob] = 0  # Exclude tokens with low probs < min_prob
                recovery_probs /= recovery_probs.sum()  # Normalize the probabilities
                
                ### 有放回抽样
                max_retries = 5
                retries = 0
                success = False

                while retries < max_retries and not success:
                    try:
                        recovery_indices = list(np.random.choice(vocab_size, size=args.n_mlm_recovery_sampling, p=recovery_probs.cpu().detach().numpy(), replace=False))
                        success = True  # 设置成功标志
                    except ValueError as e:
                        retries += 1
                        print(f"Attempt {retries} failed with error: {e}")
                        if retries >= max_retries:
                            print("Max retries reached. Skipping this iteration.")
                
                ### recovery to sequence
                if retries < max_retries:
                    for idx in [predicted_toks[i][j]] + [3+i for i in recovery_indices]:
                        recovery_seq = deepcopy(list(masked_seqs[i]))
                        recovery_tok = deepcopy(masked_toks[i])
                        
                        recovery_tok[j] = idx
                        recovery_seq[j-1] = idx_to_tok[idx]
                        
                        recovered_tok_i.append(recovery_tok)
                        recovered_sequence_i.append(''.join(recovery_seq))
                        
        recovered_sequences.extend(recovered_sequence_i)
        recovered_toks.extend(recovered_tok_i)
    return recovered_sequences, torch.LongTensor(torch.stack(recovered_toks))

def recovered_mlm_multi_tokens(masked_seqs, masked_toks, esm_logits, exclude_low_prob = False):
    # Only remain the AGCT logits
    esm_logits = esm_logits[:,:,3:7]
    # Get the predicted tokens using argmax
    predicted_toks = (esm_logits.argmax(dim=-1)+3).tolist()
    
    batch_size, seq_len, vocab_size = esm_logits.size()
    if exclude_low_prob: min_prob = 1 / vocab_size
    # Initialize an empty list to store the recovered sequences
    recovered_sequences, recovered_toks = [], []
    
    for i in range(batch_size):
        recovered_sequence_i, recovered_tok_i = [], []
        recovered_masked_num = 0
        for j in range(seq_len):
            if masked_toks[i][j] == 8:
                ### Sample M recovery sequences using the logits
                recovery_probs = torch.softmax(esm_logits[i, j], dim=-1)
                recovery_probs[predicted_toks[i][j]-3] = 0  # Exclude the most probable token
                if exclude_low_prob: recovery_probs[recovery_probs < min_prob] = 0  # Exclude tokens with low probs < min_prob
                recovery_probs /= recovery_probs.sum()  # Normalize the probabilities
                
                ### 有放回抽样
                max_retries = 5
                retries = 0
                success = False

                while retries < max_retries and not success:
                    try:
                        recovery_indices = list(np.random.choice(vocab_size, size=args.n_mlm_recovery_sampling, p=recovery_probs.cpu().detach().numpy(), replace=False))
                        success = True  # 设置成功标志
                    except ValueError as e:
                        retries += 1
                        print(f"Attempt {retries} failed with error: {e}")
                        if retries >= max_retries:
                            print("Max retries reached. Skipping this iteration.")
                
                ### recovery to sequence 
                        
                if recovered_masked_num == 0:
                    if retries < max_retries:
                        for idx in [predicted_toks[i][j]] + [3+i for i in recovery_indices]:
                            recovery_seq = deepcopy(list(masked_seqs[i]))
                            recovery_tok = deepcopy(masked_toks[i])

                            recovery_tok[j] = idx
                            recovery_seq[j] = idx_to_tok[idx]

                            recovered_tok_i.append(recovery_tok)
                            recovered_sequence_i.append(''.join(recovery_seq))
                elif recovered_masked_num > 0:
                    if retries < max_retries:
                        for idx in [predicted_toks[i][j]] + [3+i for i in recovery_indices]:
                            for recovery_seq, recovery_tok in zip(list(recovered_sequence_i), list(recovered_tok_i)): # 要在循环开始之前获取列表的副本来进行迭代。这样，在循环中即使我们修改了原始的列表，也不会影响迭代的行为。

                                recovery_seq_temp = list(recovery_seq)
                                recovery_tok[j] = idx
                                recovery_seq_temp[j] = idx_to_tok[idx]
                                
                                recovered_tok_i.append(recovery_tok)
                                recovered_sequence_i.append(''.join(recovery_seq_temp))
                
                recovered_masked_num += 1   
                
        recovered_indices = [i for i, s in enumerate(recovered_sequence_i) if '*' not in s]
        recovered_tok_i = [recovered_tok_i[i] for i in recovered_indices]
        recovered_sequence_i = [recovered_sequence_i[i] for i in recovered_indices]
        
        recovered_sequences.extend(recovered_sequence_i)
        recovered_toks.extend(recovered_tok_i)
        
        recovered_sequences, recovered_toks = remove_duplicates_double(recovered_sequences, recovered_toks)
    return recovered_sequences, torch.LongTensor(torch.stack(recovered_toks))

def mismatched_positions(s1, s2):
    # 这个函数假定两个字符串的长度相同。
    """Return the number of positions where two strings differ."""
    
    # The number of mismatches will be the sum of positions where characters are not the same
    return sum(1 for c1, c2 in zip(s1, s2) if c1 != c2)

def remove_duplicates_triple(filtered_mut_seqs, filtered_mut_probs, filtered_mut_logits):
    seen = {}
    unique_seqs = []
    unique_probs = []
    unique_logits = []
    
    for seq, prob, logit in zip(filtered_mut_seqs, filtered_mut_probs, filtered_mut_logits):
        if seq not in seen:
            unique_seqs.append(seq)
            unique_probs.append(prob)
            unique_logits.append(logit)
            seen[seq] = True

    return unique_seqs, unique_probs, unique_logits#, unique_attns

def remove_duplicates_double(filtered_mut_seqs, filtered_mut_probs):
    seen = {}
    unique_seqs = []
    unique_probs = []
    
    for seq, prob in zip(filtered_mut_seqs, filtered_mut_probs):
        if seq not in seen:
            unique_seqs.append(seq)
            unique_probs.append(prob)
            seen[seq] = True

    return unique_seqs, unique_probs


def process_batches_and_predict(model, mut_seqs): # , tok_len, batch_toks
    """
    处理批次并对每个批次进行预测，然后连接结果。

    参数:
        batch_toks (int): 每批次的总token数。
        tok_len (int): 单个序列的token长度。
        mut_toks (torch.Tensor): 要处理的序列张量。
        model (Model): 用于预测的模型实例。

    返回:
        tuple: 包含所有批次的连接概率、预测、ESM logits和logits的元组。
    """
    # 初始化列表以存储每个批次的输出
    all_mut_probs, all_mut_preds, all_mut_esm_logits, all_mut_logits = [], [], [], []

    dataset, dataloader = utrlm.generate_dataset_dataloader(['Mut']*len(mut_seqs), mut_seqs)

    with torch.no_grad():
        for (_, _, _, tok, _, _) in dataloader:
            prob, pred, esm_logit, logit = model.predict(tok.to(device), args.transform_type)

            # 将这个批次的输出添加到列表中
            all_mut_probs.append(prob.cpu().detach())
            all_mut_preds.append(pred.cpu().detach())
            all_mut_esm_logits.append(esm_logit.cpu().detach())
            all_mut_logits.append(logit.cpu().detach())
        

    # 使用 torch.cat 连接列表中的所有元素
    all_mut_probs = torch.cat(all_mut_probs, dim=0)
    all_mut_preds = torch.cat(all_mut_preds, dim=0)
    all_mut_esm_logits = torch.cat(all_mut_esm_logits, dim=0)
    all_mut_logits = torch.cat(all_mut_logits, dim=0)

    return all_mut_probs, all_mut_preds, all_mut_esm_logits, all_mut_logits#, all_mut_attns


def mutated_seq(model, wt_seq, wt_label):
    wt_dataset, wt_dataloader = utrlm.generate_dataset_dataloader(['WT'], [wt_seq])
    
    with torch.no_grad():
        for (_, _, _, wt_tok, _, _) in wt_dataloader:
            wt_prob, wt_pred, _, wt_logit = model.predict(wt_tok.to(device), args.transform_type) # logit

    tok_len = len(wt_seq)
    print(f'Wild Type: Length = ', tok_len, '\n', wt_seq)
    print(f'Wild Type: Label = {wt_label}, Y_pred = {wt_pred.item()}, Y_prob = {wt_prob.item():.2%}')
    
    pbar = tqdm(total=args.n_mut)
    mutated_seqs = []
    i = 1
    while i <= args.n_mut:
        if i == 1: seeds_ep = [wt_seq]
        seeds_next_ep, seeds_probs_next_ep, seeds_logits_next_ep = [], [], []
        for seed in seeds_ep:
            seed_dataset, seed_dataloader = utrlm.generate_dataset_dataloader(['Seed'], [seed])
            with torch.no_grad():
                for (_, seed_seq, _, seed_tok, _, _) in seed_dataloader:
                    seed_prob, seed_pred, _, seed_logit = model.predict(seed_tok.to(device), args.transform_type)
            masked_seed_tok = batch_mlm_seq(seed_tok, continuous_replace = False) ### mask seed with 1 site 
                    
            with torch.no_grad():
                _, _, seed_esm_logit, _ = model.predict(masked_seed_tok.to(device), args.transform_type)   
            masked_seed_seq = [''.join([idx_to_tok[idx] for idx in tok.tolist()]) for tok in masked_seed_tok]
            
            if args.mutation_type != 'mut_gene_editing':
                mut_seqs, mut_toks = recovered_mlm_multi_tokens(masked_seed_seq, masked_seed_tok, seed_esm_logit)
            else:
                mut_seqs, mut_toks = deepcopy(masked_seed_seq), deepcopy(masked_seed_tok)
            mut_seqs = [s[1:-1] for s in mut_seqs]
            mut_probs, mut_preds, mut_esm_logits, mut_logits = process_batches_and_predict(model, mut_seqs) 
            
            ### Filter mut_seqs that mut_prob < seed_prob and mut_prob < wild_prob
            filtered_mut_seqs = []
            filtered_mut_probs = []
            filtered_mut_logits = []
            if args.mut_by_prob:
                for z in range(len(mut_seqs)):
                    if args.mutate2stronger:
                        if mut_probs[z] >= seed_prob and mut_probs[z] >= wt_prob:
                            filtered_mut_seqs.append(mut_seqs[z])
                            filtered_mut_probs.append(mut_probs[z].cpu().detach().numpy())
                            filtered_mut_logits.append(mut_logits[z].cpu().detach().numpy())
                    else:
                        if mut_probs[z] < seed_prob and mut_probs[z] < wt_prob:
                            filtered_mut_seqs.append(mut_seqs[z])
                            filtered_mut_probs.append(mut_probs[z].cpu().detach().numpy())
                            filtered_mut_logits.append(mut_logits[z].cpu().detach().numpy())
            else:
                for z in range(len(mut_seqs)):
                    if args.mutate2stronger:
                        if mut_logits[z] >= seed_logit and mut_logits[z] >= wt_logit:
                            filtered_mut_seqs.append(mut_seqs[z])
                            filtered_mut_probs.append(mut_probs[z].cpu().detach().numpy())
                            filtered_mut_logits.append(mut_logits[z].cpu().detach().numpy())
                    else:
                        if mut_logits[z] < seed_logit and mut_logits[z] < wt_logit:
                            filtered_mut_seqs.append(mut_seqs[z])
                            filtered_mut_probs.append(mut_probs[z].cpu().detach().numpy())
                            filtered_mut_logits.append(mut_logits[z].cpu().detach().numpy())

            ### Save
            seeds_next_ep.extend(filtered_mut_seqs)
            seeds_probs_next_ep.extend(filtered_mut_probs)
            seeds_logits_next_ep.extend(filtered_mut_logits)
            seeds_next_ep, seeds_probs_next_ep, seeds_logits_next_ep = remove_duplicates_triple(seeds_next_ep, seeds_probs_next_ep, seeds_logits_next_ep)

        ### Sampling based on prob
        if len(seeds_next_ep) > args.n_sampling_designs_ep:
            seeds_probs_next_ep_norm = seeds_probs_next_ep / sum(seeds_probs_next_ep)  # Normalize the probabilities
            seeds_index_next_ep = np.random.choice(len(seeds_next_ep), args.n_sampling_designs_ep, p = seeds_probs_next_ep_norm, replace = False)
            
            seeds_next_ep = np.array(seeds_next_ep)[seeds_index_next_ep]
            seeds_probs_next_ep = np.array(seeds_probs_next_ep)[seeds_index_next_ep]
            seeds_logits_next_ep = np.array(seeds_logits_next_ep)[seeds_index_next_ep]
            
        seeds_mutated_num_next_ep = [mismatched_positions(wt_seq, s) for s in seeds_next_ep]
        
        mutated_seqs.extend(list(zip(seeds_next_ep, seeds_logits_next_ep, seeds_probs_next_ep, seeds_mutated_num_next_ep)))

        seeds_ep = seeds_next_ep
        i += 1
        pbar.update(1)
    pbar.close()

    mutated_seqs.extend([(wt_seq, wt_logit.item(), wt_prob.item(), 0)])
    mutated_seqs = sorted(mutated_seqs, key=lambda x: x[2], reverse=True)
    mutated_seqs = pd.DataFrame(mutated_seqs, columns = ['MT', 'mut_predicted_logit', 'mut_predicted_probability', 'Mut_Num']).drop_duplicates('MT')
    return mutated_seqs

    
for fold in range(10):
    state_dict = torch.load(f'{args.utrlm_modelfile}_fold{fold}.pt',
                            map_location=torch.device(device))
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k.replace('module.','')
        new_state_dict[name] = v

    utrlm_model = utrlm.CNN_linear().to(device)
    utrlm_model.load_state_dict(new_state_dict, strict = False)
    utrlm_model.eval()
    mut_res = mutated_seq(utrlm_model, args.wt_seq, wt_label = 0)
    mut_res['fold'] = fold
    if fold == 0: 
        res = deepcopy(mut_res)
    else:
        res = pd.concat([res, mut_res], axis = 0)
#     if fold == 1: break

res = res.reset_index(drop = True)
res = pd.concat([res, align_sequences(args.wt_seq, res.MT)[['Mut_Align','WT']]], axis = 1)
res.to_csv(f'../results/IRES_UTRLM_Design_{args.prefix}_Mut{args.n_mut}Sites_Stronger{args.mutate2stronger}_MutByProb{args.mut_by_prob}.csv')
 
 
