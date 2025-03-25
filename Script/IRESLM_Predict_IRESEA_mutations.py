# CUDA_VISIBLE_DEVICES=0 python3 vJun26_Predict_UTRLM_RNAFM.py --data_file vJun8_Fold0_v10.3_UTRLM1.4_FinetuneESM_lr1e-4_dr5_bos_CLS20_10folds_F0_AvgEmbFalse_BosEmbTrue_epoch100_nodes40_dropout30.5_finetuneESMTrue_finetuneLastLayerESMFalse_lr0.0001_test.fa --return_repr

import argparse
import copy
import gc
import glob
import math
import os
import random
from inspect import isfunction
from typing import List, Union

import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as save_image
from einops import rearrange
from functools import partial
# from IPython.core.debugger import set_trace
# from IPython.display import display, Image
# from livelossplot import PlotLosses
# from livelossplot.outputs import MatplotlibPlot
from PIL import Image
from scipy.optimize import fsolve, minimize
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr, kl_div
from scipy.misc import derivative
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from torch import einsum
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules.activation import ReLU
from torchmetrics.functional import kl_divergence
from tqdm import tqdm, tqdm_notebook
import itertools
import RNA
from Bio import SeqIO

import warnings
warnings.filterwarnings('ignore')

import vMay7_UTRLM_Predictor as utrlm
import vMay7_RNAFM_Predictor as rnafm
from collections import OrderedDict
# Set up any necessary specific library configurations
matplotlib.use('Agg')  # For matplotlib to work in headless mode
sns.set(style="whitegrid")  # Setting the seaborn style
# %matplotlib inline

parser = argparse.ArgumentParser(description="Training configuration for UNET with time warping.")
parser.add_argument('--device', type=str, default='0', help='Prefix for the run')
parser.add_argument('--prefix', type=str, default='vJun26', help='Prefix for the run')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--data_file', type = str, default = '/oak/stanford/scg/lab_congle/yanyichu/yanyichu/DeepCIP/data/experimentally_validated_circRNA_IRES_n21.fa')

parser.add_argument('--global_seed', type=int, default=42, help='Global seed for randomness')
parser.add_argument('--utrlm_modelfile', type = str, default = 'IRES_UTRLM_best_model')
parser.add_argument('--rnafm_modelfile', type = str, default = 'IRES_RNAFM_best_model')
parser.add_argument('--return_repr', action='store_true')
parser.add_argument('--return_attn', action='store_true')
parser.add_argument('--column_name', type=str, default='MT', help='The column name containing sequences.')


args = parser.parse_args()
print(args)

BATCH_SIZE = args.batch_size
GLOBAL_SEED = args.global_seed

utrlm_folds = 10
rnafm_folds = 10
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
print(device)
prefix = f'{args.prefix}_{args.data_file.split("/")[-1].split(".")[0]}'
print(prefix)


def seed_everything(seed=GLOBAL_SEED):
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

if '.fa' in args.data_file:
    seqs = []
    for record in SeqIO.parse(args.data_file, 'fasta'):
        seqs.append(str(record.seq))
else:
    df = pd.read_csv(args.data_file)#.dropna()
    seqs = list(df[args.column_name])
        
def process_and_evaluate_sequences(seqs):
    
    
    # UTR-LM
    utrlm_dataset, utrlm_dataloader = utrlm.generate_dataset_dataloader(['Predict']*len(seqs), seqs)

    utrlm_results = pd.DataFrame()
    for fold in range(utrlm_folds):
        print(f'-----UTRLM Fold{fold}-----')
        state_dict = torch.load(f'../models/{args.utrlm_modelfile}_fold{fold}.pt',
                                map_location=torch.device(device))
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k.replace('module.','')
            new_state_dict[name] = v

        utrlm_model = utrlm.CNN_linear().to(device)
        utrlm_model.load_state_dict(new_state_dict, strict = False)
        utrlm_model.eval()
        utrlm_pred = utrlm.predict_step(utrlm_dataloader, utrlm_model, fold, return_repr = args.return_repr, return_attn = args.return_attn)
        
        utrlm_repr = utrlm_pred.get('x_repr')
        utrlm_attn = utrlm_pred.get('x_attn')
        # 保存 x_repr 和 x_attn
        if utrlm_repr is not None: 
            print(utrlm_repr.shape)
            np.save(f'../results/IRES_UTRLM_repr_{prefix}_fold{fold}.npy', utrlm_repr)
        if utrlm_attn is not None: 
            print(utrlm_attn.shape)
            np.save(f'../results/IRES_UTRLM_attn_{prefix}_fold{fold}.npy', utrlm_attn)

        utrlm_result = utrlm_pred['df']
        utrlm_result.set_index('sequence', inplace=True)
        
        if fold == 0:
            utrlm_results = utrlm_result
        else:
            utrlm_results = pd.concat([utrlm_results, utrlm_result], axis=1)
#         break
    utrlm_prob_cols = [f'Prob_U{i}' for i in range(utrlm_folds)]
    utrlm_pred_cols = [f'Pred_U{i}' for i in range(utrlm_folds)]
    utrlm_results['Prob_U_Mean'] = utrlm_results[utrlm_prob_cols].mean(axis = 1)
    utrlm_results['Pred_U_Mean'] = utrlm_results[utrlm_pred_cols].mean(axis = 1)
#     print(utrlm_results.Prob_U0)
    # RNA-FM
    rnafm_dataset, rnafm_dataloader = rnafm.generate_dataset_dataloader(['Predict']*len(seqs), seqs)

    rnafm_results = pd.DataFrame()
    for fold in range(rnafm_folds):
        print(f'-----RNAFM Fold{fold}-----')
        state_dict = torch.load(f'../models/{args.rnafm_modelfile}_fold{fold}.pt', map_location=torch.device(device))
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k.replace('module.','')
            new_state_dict[name] = v

        rnafm_model = rnafm.RNAFM_linear().to(device)
        rnafm_model.load_state_dict(new_state_dict, strict = True)
        rnafm_model.eval()
        rnafm_pred = rnafm.predict_step(rnafm_dataloader, rnafm_model, fold, return_repr = args.return_repr)
        
        rnafm_repr = rnafm_pred.get('x_repr')
        rnafm_attn = rnafm_pred.get('x_attn')
        
        # 保存 x_repr 和 x_attn
        if rnafm_repr is not None: 
            print(rnafm_repr.shape)
            np.save(f'../results/IRES_RNAFM_repr_{prefix}_fold{fold}.npy', rnafm_repr)
        if rnafm_attn is not None: 
            print(rnafm_attn.shape)
            np.save(f'../results/IRES_RNAFM_attn_{prefix}_fold{fold}.npy', rnafm_attn)
            
        
        rnafm_result = rnafm_pred['df']
        rnafm_result.set_index('sequence', inplace=True)
        if fold == 0:
            rnafm_results = rnafm_result
        else:
            rnafm_results = pd.concat([rnafm_results, rnafm_result], axis=1)
#         break
    rnafm_results.index = rnafm_results.index.str.replace('U', 'T')
    rnafm_prob_cols = [f'Prob_R{i}' for i in range(rnafm_folds)]
    rnafm_pred_cols = [f'Pred_R{i}' for i in range(rnafm_folds)]
    rnafm_results['Prob_R_Mean'] = rnafm_results[rnafm_prob_cols].mean(axis = 1)
    rnafm_results['Pred_R_Mean'] = rnafm_results[rnafm_pred_cols].mean(axis = 1)

    # IRES-LM
    ireslm_results = pd.merge(utrlm_results, 
                              rnafm_results, 
                              on = ['sequence'])
    ireslm_results['Prob_Mean'] = ireslm_results[utrlm_prob_cols + rnafm_prob_cols].mean(axis = 1)
    ireslm_results['Pred_Mean'] = ireslm_results[utrlm_pred_cols + rnafm_pred_cols].mean(axis = 1)
    ireslm_results['length'] = [len(s) for s in ireslm_results.index]
    
    ireslm_results = ireslm_results[['length', 'Prob_Mean', 'Pred_Mean', 
                                 'Pred_U_Mean', 'Pred_R_Mean', 'Prob_U_Mean', 'Prob_R_Mean']
                               + utrlm_prob_cols + rnafm_prob_cols + utrlm_pred_cols + rnafm_pred_cols]
    
    ### Save
    ireslm_results.to_csv(f'../results/IRESLM_{prefix}.csv')

    return ireslm_results

ireslm_results = process_and_evaluate_sequences(seqs)
ireslm_results