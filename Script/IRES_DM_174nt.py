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
from IPython.core.debugger import set_trace
from IPython.display import display, Image
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
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
from polyleven import levenshtein
from scipy.stats import ks_2samp,kstest,ttest_ind, mannwhitneyu, norm
from cliffs_delta import cliffs_delta
import logomaker
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

import UTRLM_Predictor as utrlm
import RNAFM_Predictor as rnafm
from collections import OrderedDict
# Set up any necessary specific library configurations
matplotlib.use('Agg')  # For matplotlib to work in headless mode
sns.set(style="whitegrid")  # Setting the seaborn style

parser = argparse.ArgumentParser(description="Training configuration for UNET with time warping.")
parser.add_argument('--device', type=str, default='0', help='Prefix for the run')
parser.add_argument('--prefix', type=str, default='vMay23_IRESDM', help='Prefix for the run')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train for')
parser.add_argument('--image_size', type=int, default=176, help='Number of base pairs for the motif')
parser.add_argument('--only_ires', action='store_true', help='Use time warping to sample noise smartly in an active learning setting')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--timesteps', type=int, default=50, help='Timesteps for diffusion')

parser.add_argument('--beta_scheduler', type=str, default='linear', help='Prefix for the run')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
parser.add_argument('--global_seed', type=int, default=42, help='Global seed for randomness')
parser.add_argument('--ema_beta', type=float, default=0.995, help='Learning rate for the optimizer')
parser.add_argument('--n_steps', type=int, default=10, help='Number of steps for standard UNET training before time warping')
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples for training and evaluation')
parser.add_argument('--save_and_sample_every', type=int, default=1, help='Epoch interval for saving and comparing metrics')
parser.add_argument('--epochs_loss_show', type=int, default=5, help='Epoch interval to show loss')
parser.add_argument('--time_warping', type=bool, default=True, help='Use time warping to sample noise smartly in an active learning setting')
parser.add_argument('--truncate_remain_right', action='store_false', help='Use time warping to sample noise smartly in an active learning setting')
parser.add_argument('--utrlm_modelfile', type = str, default = '../models/IRES_UTRLM_best_model')
parser.add_argument('--rnafm_modelfile', type = str, default = '../models/IRES_RNAFM_best_model')
parser.add_argument('--cls_model', type=str, choices=['rnafm', 'utrlm', 'both', ''], default='both', 
                    help='Choose to use RNA-FM, UTRL-M, or both models.')
args = parser.parse_args()
print(args)

NUCLEOTIDES = ['A', 'C', 'T', 'G']
CHANNELS = 1
RESNET_BLOCK_GROUPS = 4

IMAGE_SIZE = args.image_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
GLOBAL_SEED = args.global_seed
N_STEPS = args.n_steps
TIMESTEPS = args.timesteps
N_SAMPLES = args.n_samples
SAVE_AND_SAMPLE_EVERY = args.save_and_sample_every
TIME_WARPING = args.time_warping
EPOCHS_LOSS_SHOW = args.epochs_loss_show

utrlm_folds = 10
rnafm_folds = 10
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
print(device)
prefix = f"{args.prefix}_Epoch{EPOCHS}_L{IMAGE_SIZE}_{['CutRight', 'CutLeft'][args.truncate_remain_right]}_{['woIRES', 'wIRES'][args.only_ires]}_Batch{BATCH_SIZE}_TimeSteps{TIMESTEPS}_{args.beta_scheduler}BETA_lr{str(LEARNING_RATE)}_CLS{args.cls_model}"
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

def calculate_class_weight(labels):
    """
    计算类别权重，基于类别在标签中的出现频率。
    
    参数:
    - labels (Tensor): 包含类别标签的张量
    
    返回:
    - class_weight (Tensor): 每个类别的权重
    """
    # 计算每个类别的出现次数
    class_counts = torch.bincount(labels)
    if class_counts.numel() == 1:
        return torch.tensor([1.0])
    # 避免除以零
    class_counts = class_counts.float()
    # 计算总样本数
    total_samples = labels.numel()
    # 计算每个类别的频率
    class_freq = class_counts / total_samples
    # 计算权重：频率的倒数
    class_weight = 1.0 / class_freq
    # 归一化权重，使得它们的总和等于类别数
    class_weight = class_weight * class_weight.numel() / class_weight.sum()
    
    return class_weight

data = pd.read_csv('../Data/v2_dataset_with_unified_stratified_shuffle_train_test_split.csv', index_col=0)
data = data[data.fold == 0]
data = data[data['Sequence'].str.len() == 174].sample(frac=1, random_state=GLOBAL_SEED).reset_index(drop=True)#.iloc[:1000]

# 正确的训练/验证分割
train_num = int(len(data) * 1)
train_data = data.iloc[:train_num]
val_data = data.iloc[:train_num]#[train_num:] 

if args.only_ires:
    TOTAL_CLASS_NUMBER = 1
    train_data = train_data[train_data.IRES_class_600 == 1].reset_index(drop=True)
    val_data = val_data[val_data.IRES_class_600 == 1].reset_index(drop=True)  # 验证集也要处理
    cell_types = [1]
    class_weight = None
else:
    TOTAL_CLASS_NUMBER = 2
    cell_types = [0, 1]
    class_weight = calculate_class_weight(torch.LongTensor(train_data.IRES_class_600.values))
print(f"174nt Train = {len(train_data)}, Val = {len(val_data)}")

def load_all_cls_models(device, args, utrlm_folds=10, rnafm_folds=10):
    
    utrlm_models = None
    rnafm_models = None
    
    print("Loading UTRLM models...")
    utrlm_models = []
    for fold in range(utrlm_folds):
        state_dict = torch.load(f'{args.utrlm_modelfile}_fold{fold}.pt', map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.','')
            new_state_dict[name] = v
        
        model = utrlm.CNN_linear().to(device)
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        # for param in model.parameters():
        #     param.requires_grad = False  # 冻结参数，节省内存
        utrlm_models.append(model)
    print(f"Loaded {len(utrlm_models)} UTRLM models")
    

    print("Loading RNAFM models...")
    rnafm_models = []
    for fold in range(rnafm_folds):
        state_dict = torch.load(f'{args.rnafm_modelfile}_fold{fold}.pt', map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.','')
            new_state_dict[name] = v
        
        model = rnafm.RNAFM_linear().to(device)
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        # for param in model.parameters():
        #     param.requires_grad = False  # 冻结参数，节省内存
        rnafm_models.append(model)
    print(f"Loaded {len(rnafm_models)} RNAFM models")
    
    return utrlm_models, rnafm_models
    
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
        
# Scheduler
class BetaScheduler:
    def __init__(self, timesteps, scheduler_type, **kwargs):
        self.timesteps = timesteps
        self.scheduler_type = scheduler_type
        self.kwargs = kwargs

    def cosine_beta_schedule(self):
        s = self.kwargs.get('s', 0.008)
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def linear_beta_schedule(self):
        beta_end = self.kwargs.get('beta_end', 0.005)
        beta_start = 0.0001
        return torch.linspace(beta_start, beta_end, self.timesteps)

    def quadratic_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start**0.5, beta_end**0.5, self.timesteps) ** 2

    def sigmoid_beta_schedule(self):
        beta_start = 0.001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, self.timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    def get_betas(self):
        if self.scheduler_type == 'cosine':
            return self.cosine_beta_schedule()
        elif self.scheduler_type == 'linear':
            return self.linear_beta_schedule()
        elif self.scheduler_type == 'quadratic':
            return self.quadratic_beta_schedule()
        elif self.scheduler_type == 'sigmoid':
            return self.sigmoid_beta_schedule()
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

beta_scheduler = BetaScheduler(timesteps=TIMESTEPS, scheduler_type=args.beta_scheduler, beta_end=0.2)
betas = beta_scheduler.get_betas()
# define alphas
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) 

# From Seq_Diff, 计算的是 sqrt(1.0 / alphas_cumprod - 1)
sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1) 

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # 预测噪声
    model_output = model(x, time=t, classes=None)
    
    # 计算均值
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        
        result = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        return result
        
@torch.no_grad()
def p_sample_guided(model, x, classes, t, t_index, context_mask, cond_weight=0.0):
    # adapted from: https://openreview.net/pdf?id=qw8AKxfYbI
    # print (classes[0])
    batch_size = x.shape[0]
    # double to do guidance with
    t_double = t.repeat(2)
    x_double = x.repeat(2, 1, 1, 1)
    betas_t = extract(betas, t_double, x_double.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t_double, x_double.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t_double, x_double.shape)

    # classifier free sampling interpolates between guided and non guided using `cond_weight`
    classes_masked = classes * context_mask
    classes_masked = classes_masked.type(torch.long)
    # print ('class masked', classes_masked)
    preds = model(x_double, time=t_double, classes=classes_masked)
    eps1 = (1 + cond_weight) * preds[:batch_size]
    eps2 = cond_weight * preds[batch_size:]
    x_t = eps1 - eps2

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t[:batch_size] * (
        x - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
    )
    # 在返回之前也需要确保首尾为0
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        
        result = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        return result


# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(model, classes, shape, cond_weight):
    device = next(model.parameters()).device
    b = shape[0]

    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    
    imgs = []

    if classes is not None:
        n_sample = classes.shape[0]
        context_mask = torch.ones_like(classes).to(device)
        # make 0 index unconditional
        # double the batch
        classes = classes.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 0.0  # makes second half of batch context free
        sampling_fn = partial(p_sample_guided, classes=classes, cond_weight=cond_weight, context_mask=context_mask)
    else:
        sampling_fn = partial(p_sample)

    for i in tqdm(reversed(range(0, TIMESTEPS)), desc='sampling loop time step', total=TIMESTEPS):
        img = sampling_fn(model, x=img, t=torch.full((b,), i, device=device, dtype=torch.long), t_index=i)
        imgs.append(img.cpu().numpy())
        
    return imgs


@torch.no_grad()
def sample(model, image_size, classes=None, batch_size=16, channels=3, cond_weight=0):
    return p_sample_loop(model, classes=classes, shape=(batch_size, channels, len(NUCLEOTIDES), image_size), cond_weight=cond_weight)

def sampling_to_metric(model_best, number_of_samples=20, specific_group=False, group_number=None, cond_weight_to_metric=0):
    """
    This function encapsulates the logic of sampling from the trained model in order to generate counts of the motifs.
    The reasoning is that we are interested only in calculating the evaluation metric
    for the count of occurances and not the nucleic acids themselves.
    """
    seq_final = []
    total_count = 0
    for n_a in range(number_of_samples):
        
        if specific_group:
            sampled = torch.from_numpy(np.array([group_number] * BATCH_SIZE))
            print(f'********** Specific Class = {group_number} **********')
        else:
            sampled = torch.from_numpy(np.random.choice(cell_types, BATCH_SIZE))

        if not args.only_ires:
            random_classes = sampled.float().cuda()
        else:
            random_classes = None
        
        sampled_images = sample(
            model_best,
            classes=random_classes,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            channels=1,
            cond_weight=cond_weight_to_metric,
        )
        
        for n_b, x in enumerate(sampled_images[-1]):
            # 只取中间174个位置（忽略首尾padding）
            seq = ''.join([NUCLEOTIDES[s] for s in np.argmax(x.reshape(4, 176), axis=0)[1:-1]])
            total_count += 1
            
            # 直接检查是否都是ACGT
            if len(seq) == 174 and all(n in ['A', 'C', 'T', 'G'] for n in seq):
                seq_final.append(seq)
                    
    return list(set(seq_final)), total_count

def q_sample(x_start, t, noise=None):
    """
    Forward pass with noise.
    """
    if noise is None:
        noise = torch.randn_like(x_start)
        
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    
def predict_xstart_from_eps(x_t, t, eps):
    assert x_t.shape == eps.shape
    return (
        extract(sqrt_recip_alphas, t, x_t.shape) * x_t
        - extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    )

def calculate_weighted_nll_loss_with_mask(x_start, pred_x_start, seq_mask, sample_weight=None):
    pred_x_start = pred_x_start.squeeze(1)
    x_start = x_start.argmax(dim=2).squeeze(1)
    
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    nll_loss = loss_fct(pred_x_start, x_start)
    
    # 应用序列mask（去掉第一维）
    seq_mask = seq_mask.squeeze(1).squeeze(1)
    nll_loss = nll_loss * seq_mask
    
    if sample_weight is not None:
        sample_weight = sample_weight.unsqueeze(1).expand_as(nll_loss)
        nll_loss = nll_loss * sample_weight
        total_loss = nll_loss.sum() / (sample_weight * seq_mask).sum()
    else:
        total_loss = nll_loss.sum() / seq_mask.sum()
    
    return total_loss
    
def p_losses(denoise_model, x_start, t, classes, noise=None, loss_type="l1", 
             p_uncond=0.1, sample_weight=None, class_weight=None,
             utrlm_models=None, rnafm_models=None):
    device = x_start.device
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # 创建mask
    seq_mask = create_seq_mask(x_start.shape[0], x_start.shape[-1]).to(device)
    
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    
    if not args.only_ires:
        context_mask = torch.bernoulli(torch.zeros(classes.shape[0]) + (1 - p_uncond)).to(device)
        classes = classes * context_mask
        classes = classes.type(torch.long)
        predicted_noise = denoise_model(x_noisy, t, classes)
    else:
        predicted_noise = denoise_model(x_noisy, t, None)
    
    # 应用mask到噪声
    predicted_noise = predicted_noise * seq_mask
    noise = noise * seq_mask
    
    ##### Diffusion Loss
    if loss_type == 'l1':
        dif_loss = F.l1_loss(predicted_noise, noise, reduction='none')
    elif loss_type == 'l2':
        dif_loss = F.mse_loss(predicted_noise, noise, reduction='none')
    elif loss_type == "huber":
        dif_loss = F.smooth_l1_loss(predicted_noise, noise, reduction='none')
    else:
        raise NotImplementedError("Unsupported loss type provided.")

    # Apply the sequence mask to the loss if provided
    if seq_mask is not None:
        dif_loss = dif_loss * seq_mask

    # Aggregate the loss
    dif_loss = dif_loss.mean()
    
    # NLL loss也需要考虑mask
    pred_x_start = predict_xstart_from_eps(x_noisy, t, predicted_noise)
    nll_loss = calculate_weighted_nll_loss_with_mask(x_start, pred_x_start, seq_mask, sample_weight)
    
    # 初始化所有loss值为数值类型
    utrlm_loss_tensor = torch.tensor(0.0, device=device, requires_grad=True)
    rnafm_loss_tensor = torch.tensor(0.0, device=device, requires_grad=True)
    
    #######
    if args.cls_model in ['utrlm', 'rnafm', 'both'] and args.cls_model != '':
        ##### CLS Loss
        pred_x_start_logits = pred_x_start.squeeze(1)  # [batch, 4, seq_len]
    
        # 在GPU上获取argmax结果
        pred_x_start_indices = torch.argmax(pred_x_start_logits[:, :, 1:-1], dim=1)  # [batch, 174]
        
        pred_seqs, remain_idx = [], []
        
        # 在GPU上处理，但逻辑保持一致
        for i in range(pred_x_start_indices.shape[0]):
            # 获取当前序列的核苷酸索引（在GPU上）
            seq_indices = pred_x_start_indices[i]  # [174]
            
            # 转换为序列字符串
            seq = ''.join([NUCLEOTIDES[idx.item()] for idx in seq_indices])
            
            # 直接检查长度和核苷酸（保持原逻辑）
            if len(seq) == 174 and all(n in ['A', 'C', 'T', 'G'] for n in seq):
                pred_seqs.append(seq)
                remain_idx.append(i)
                
        if len(remain_idx) > 0:    
            classes = classes[remain_idx]
            
            # 只在需要且模型可用时计算UTRLM loss
            if args.cls_model in ['utrlm', 'both'] and utrlm_models is not None:
                utrlm_dataset, utrlm_dataloader = utrlm.generate_dataset_dataloader(
                    classes.detach().cpu().tolist(), pred_seqs
                )
                
                utrlm_results = pd.DataFrame()
                
                for fold, model in enumerate(utrlm_models):
                    utrlm_result = utrlm.predict_step(utrlm_dataloader, model, fold)
                    utrlm_result = utrlm_result['df']
                    utrlm_result.set_index('sequence', inplace=True)
                    if fold == 0:
                        utrlm_results = utrlm_result
                    else:
                        utrlm_results = pd.concat([utrlm_results, utrlm_result], axis=1)
                
                utrlm_prob_cols = [f'Prob_U{i}' for i in range(len(utrlm_models))]
                utrlm_results['Prob_U_Mean'] = utrlm_results[utrlm_prob_cols].mean(axis=1)
                
                utrlm_prob = utrlm_results['Prob_U_Mean']
                utrlm_prob = torch.tensor(utrlm_prob.values, device=device, dtype=torch.float32, requires_grad=True)

                utrlm_loss_tensor = nn.BCELoss()(utrlm_prob, classes.float())
            
            # 只在需要且模型可用时计算RNAFM loss
            if args.cls_model in ['rnafm', 'both'] and rnafm_models is not None:
                rnafm_dataset, rnafm_dataloader = rnafm.generate_dataset_dataloader(
                    classes.detach().cpu().tolist(), pred_seqs
                )
                
                rnafm_results = pd.DataFrame()
                
                for fold, model in enumerate(rnafm_models):
                    rnafm_result = rnafm.predict_step(rnafm_dataloader, model, fold)
                    rnafm_result = rnafm_result['df']
                    rnafm_result.set_index('sequence', inplace=True)
                    if fold == 0:
                        rnafm_results = rnafm_result
                    else:
                        rnafm_results = pd.concat([rnafm_results, rnafm_result], axis=1)
                
                rnafm_prob_cols = [f'Prob_R{i}' for i in range(len(rnafm_models))]
                rnafm_results['Prob_R_Mean'] = rnafm_results[rnafm_prob_cols].mean(axis=1)
                
                rnafm_prob = rnafm_results['Prob_R_Mean']
                rnafm_prob = torch.tensor(rnafm_prob.values, device=device, dtype=torch.float32, requires_grad=True)
                rnafm_loss_tensor = nn.BCELoss()(rnafm_prob, classes.float())
            
    if args.cls_model == 'utrlm':
        return nll_loss, dif_loss, utrlm_loss_tensor, utrlm_loss_tensor, rnafm_loss_tensor
    elif args.cls_model == 'rnafm':
        return nll_loss, dif_loss, rnafm_loss_tensor, utrlm_loss_tensor, rnafm_loss_tensor
    elif args.cls_model == 'both':
        cls_loss = utrlm_loss_tensor + rnafm_loss_tensor
        return nll_loss, dif_loss, cls_loss, utrlm_loss_tensor, rnafm_loss_tensor
    elif args.cls_model == '':
        return nll_loss, dif_loss, 0, utrlm_loss_tensor, rnafm_loss_tensor


def create_seq_mask(batch_size, seq_length=176):
    """创建序列mask，第一个和最后一个位置为0，其他为1"""
    mask = torch.ones(batch_size, 1, 1, seq_length)
    mask[:, :, :, 0] = 0    # 第一个位置
    mask[:, :, :, -1] = 0   # 最后一个位置
    return mask
    
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, class_weight=None, transform=None):
        self.labels = labels
        self.nucleotides = NUCLEOTIDES  # ['A', 'C', 'T', 'G']
        self.max_seq_len = IMAGE_SIZE   # 176
        self.transform = transform
        self.images = []
        self.class_weight = class_weight

        for seq in sequences:
            encoded_img = self.one_hot_encode(seq)
            self.images.append(encoded_img)

        self.images = np.array([x.T.tolist() for x in self.images])

    def one_hot_encode(self, seq):
        assert len(seq) == 174  # 输入应该是174nt
        seq_array = np.zeros((176, 4))  # 176长度，4个核苷酸
        
        # 第一个和最后一个位置保持为0（padding）
        for i, nucleotide in enumerate(seq):
            if nucleotide in self.nucleotides:
                seq_array[i+1, self.nucleotides.index(nucleotide)] = 1
        
        return seq_array

    def __len__(self):  # 添加这个方法
        """
        Denotes the total number of samples.
        """
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        
        if self.class_weight is not None: 
            sample_weight = self.class_weight[label]
        else:
            sample_weight = torch.tensor([1.0])
        
        return image, label, sample_weight
        
def random_data(length=174, size=1000):
    """
    生成固定长度174nt的随机序列
    
    参数:
        length (int): 实际序列长度，默认174
        size (int): 生成的序列数量
    返回:
        list: 生成的序列列表（长度174）
    """
    samples = []
    for _ in range(size):
        # 生成174nt的随机序列
        random_seq = ''.join(random.choice(['A', 'T', 'C', 'G']) for _ in range(length))
        samples.append(random_seq)
    
    return samples
    
def seq_to_fasta(seq_list, outfilename):
    """
    将序列列表写入一个FASTA格式的文件。

    参数:
    seq_list : list
        DNA或蛋白质序列的列表。
    outfilename : str
        输出文件的名称。
    """
    # 打开文件准备写入
    with open(outfilename, 'w') as file:
        # 遍历序列列表，每个序列分配一个唯一的标识符
        for index, seq in enumerate(seq_list):
            # 写入FASTA格式的头部
            file.write(f">seq_{index}_{len(seq)}\n{seq}\n")

def process_and_evaluate_sequences(train_data, model_best, class_=1, seqlogo_max_len=IMAGE_SIZE, 
                                  kmer=False, hamming=False, gc=True, mfe=False, ss_hamming=False,
                                  utrlm_models=None, rnafm_models=None):  # 添加预加载模型参数
    """
    修改后的评估函数，接收预加载的模型
    """
    # 选择真实序列（已经是174nt，没有S和E）
    real_seqs = train_data[train_data.IRES_class_600 == class_]['Sequence'].values.tolist()
    
    # 生成随机序列（也是174nt）
    random_seqs = random_data(length=174, size=len(real_seqs))
    
    # Model evaluation and sequence generation
    model_best.eval()
    number_of_batches = math.ceil(len(real_seqs) / BATCH_SIZE)
    gen_seqs, total_count = sampling_to_metric(model_best, number_of_samples=number_of_batches,  
                                              specific_group=True, group_number=class_)
    
    valid_count = len(gen_seqs)
    print(f'MaxLength = {IMAGE_SIZE} | Num: Correct_Generate = {valid_count}/{total_count} | Real = {len(real_seqs)} | Random = {len(random_seqs)}')
    
    # Saving generated sequences to a FASTA file
    seq_to_fasta(gen_seqs, f'../results/{prefix}_Gene_Class{class_}_num{valid_count}_ep{best_ep}.fasta')
    seq_to_fasta(real_seqs, f'../results/{prefix}_Real_Class{class_}_num{len(real_seqs)}_ep{best_ep}.fasta')
    seq_to_fasta(random_seqs, f'../results/{prefix}_Rand_Class{class_}_num{len(random_seqs)}_ep{best_ep}.fasta')
    
    # 只有在有生成序列时才进行评估
    if len(gen_seqs) == 0:
        print("No valid sequences generated, skipping evaluation")
        return gen_seqs, real_seqs, random_seqs
    
    # UTR-LM评估 - 使用预加载的模型
    utrlm_results = pd.DataFrame()
    if utrlm_models is not None:  # 只在模型可用时运行
        utrlm_dataset, utrlm_dataloader = utrlm.generate_dataset_dataloader(
            [f'Generated_Class{class_}']*len(gen_seqs), gen_seqs
        )
        
        with torch.no_grad():  # 评估时不需要梯度
            for fold, model in enumerate(utrlm_models):
                utrlm_result = utrlm.predict_step(utrlm_dataloader, model, fold)
                utrlm_result = utrlm_result['df']
                utrlm_result.set_index('sequence', inplace=True)
                if fold == 0:
                    utrlm_results = utrlm_result
                else:
                    utrlm_results = pd.concat([utrlm_results, utrlm_result], axis=1)
        
        utrlm_prob_cols = [f'Prob_U{i}' for i in range(len(utrlm_models))]
        utrlm_pred_cols = [f'Pred_U{i}' for i in range(len(utrlm_models))]
        utrlm_results['Prob_U_Mean'] = utrlm_results[utrlm_prob_cols].mean(axis=1)
        utrlm_results['Pred_U_Mean'] = utrlm_results[utrlm_pred_cols].mean(axis=1)
    
    # RNA-FM评估 - 使用预加载的模型
    rnafm_results = pd.DataFrame()
    if rnafm_models is not None:  # 只在模型可用时运行
        rnafm_dataset, rnafm_dataloader = rnafm.generate_dataset_dataloader(
            [f'Generated_Class{class_}']*len(gen_seqs), gen_seqs
        )
        
        with torch.no_grad():  # 评估时不需要梯度
            for fold, model in enumerate(rnafm_models):
                rnafm_result = rnafm.predict_step(rnafm_dataloader, model, fold)
                rnafm_result = rnafm_result['df']
                rnafm_result.set_index('sequence', inplace=True)
                if fold == 0:
                    rnafm_results = rnafm_result
                else:
                    rnafm_results = pd.concat([rnafm_results, rnafm_result], axis=1)
        
        rnafm_results.index = rnafm_results.index.str.replace('U', 'T')
        rnafm_prob_cols = [f'Prob_R{i}' for i in range(len(rnafm_models))]
        rnafm_pred_cols = [f'Pred_R{i}' for i in range(len(rnafm_models))]
        rnafm_results['Prob_R_Mean'] = rnafm_results[rnafm_prob_cols].mean(axis=1)
        rnafm_results['Pred_R_Mean'] = rnafm_results[rnafm_pred_cols].mean(axis=1)
    
    # IRES-LM合并结果 - 根据可用的模型调整
    if utrlm_models is not None and rnafm_models is not None:
        # 两个模型都可用
        ireslm_results = pd.merge(utrlm_results, rnafm_results, on=['sequence'])
        ireslm_results['Prob_Mean'] = ireslm_results[utrlm_prob_cols + rnafm_prob_cols].mean(axis=1)
        ireslm_results['Pred_Mean'] = ireslm_results[utrlm_pred_cols + rnafm_pred_cols].mean(axis=1)
    elif utrlm_models is not None:
        # 只有UTRLM可用
        ireslm_results = utrlm_results.copy()
        ireslm_results['Prob_Mean'] = ireslm_results['Prob_U_Mean']
        ireslm_results['Pred_Mean'] = ireslm_results['Pred_U_Mean']
        rnafm_prob_cols = []
        rnafm_pred_cols = []
    elif rnafm_models is not None:
        # 只有RNAFM可用
        ireslm_results = rnafm_results.copy()
        ireslm_results['Prob_Mean'] = ireslm_results['Prob_R_Mean']
        ireslm_results['Pred_Mean'] = ireslm_results['Pred_R_Mean']
        utrlm_prob_cols = []
        utrlm_pred_cols = []
    else:
        # 没有模型可用，返回
        print("No evaluation models available")
        return gen_seqs, real_seqs, random_seqs
    
    ireslm_results['length'] = [len(s) for s in ireslm_results.index]
    
    # 构建列名列表，只包含存在的列
    columns = ['length', 'Prob_Mean', 'Pred_Mean']
    if 'Pred_U_Mean' in ireslm_results.columns:
        columns.append('Pred_U_Mean')
    if 'Pred_R_Mean' in ireslm_results.columns:
        columns.append('Pred_R_Mean')
    if 'Prob_U_Mean' in ireslm_results.columns:
        columns.append('Prob_U_Mean')
    if 'Prob_R_Mean' in ireslm_results.columns:
        columns.append('Prob_R_Mean')
    
    columns.extend(utrlm_prob_cols + rnafm_prob_cols + utrlm_pred_cols + rnafm_pred_cols)
    
    ireslm_results = ireslm_results[columns]
    
    if 'Prob_Mean' in ireslm_results.columns:
        print(f"Average IRES probability: {ireslm_results['Prob_Mean'].mean():.4f}")
        print(f"High confidence sequences (>0.9): {(ireslm_results['Prob_Mean'] > 0.9).sum()}")
        ireslm_results = ireslm_results.sort_values('Prob_Mean', ascending = False)
    
    ireslm_results.to_csv(f'../results/IRESLM_{prefix}_Gene_Class{class_}_correctNum{valid_count}_total{total_count}_ep{best_ep}.csv')
    
    return gen_seqs, real_seqs, random_seqs
    

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def l2norm(t):
    return F.normalize(t, dim=-1)


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# Building blocks of UNET


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# Building blocks of UNET, positional embeds


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Building blocks of UNET, building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


# Building blocks of UNET, residual part


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


# Additional code to the https://github.com/lucidrains/bit-diffusion/blob/main/bit_diffusion/bit_diffusion.py


class ResnetBlockClassConditioned(ResnetBlock):
    def __init__(self, dim, dim_out, *, num_classes, class_embed_dim, time_emb_dim=None, groups=8):
        super().__init__(dim=dim + class_embed_dim, dim_out=dim_out, time_emb_dim=time_emb_dim, groups=groups)
        self.class_mlp = EmbedFC(num_classes, class_embed_dim)

    def forward(self, x, time_emb=None, c=None):
        emb_c = self.class_mlp(c)
        emb_c = emb_c.view(*emb_c.shape, 1, 1)
        emb_c = emb_c.expand(-1, -1, x.shape[-2], x.shape[-1])
        x = torch.cat([x, emb_c], axis=1)

        return super().forward(x, time_emb)


# Building blocks of UNET, attention part


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class Unet(nn.Module):
    """
    Refer to the main paper for the architecture details https://arxiv.org/pdf/2208.04202.pdf
    """

    def __init__(
        self,
        dim,
        init_dim=IMAGE_SIZE,
        dim_mults=(1, 2, 4),
        channels=CHANNELS,
        resnet_block_groups=8,
        learned_sinusoidal_dim=18,
        num_classes=2,
        class_embed_dim=3,
    ):
        super().__init__()

        self.channels = channels
        # if you want to do self conditioning uncomment this
        # input_channels = channels * 2
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, (7, 7), padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.scale_factor = [2,2]
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in, self.scale_factor[ind]) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)
        print('final', dim, channels, self.final_conv)

    # Additional code to the https://github.com/lucidrains/bit-diffusion/blob/main/bit_diffusion/bit_diffusion.py mostly in forward method.

    def forward(self, x, time, classes):
        
        x = self.init_conv(x)
        r = x.clone()
        
        t_start = self.time_mlp(time)
        t_mid = t_start.clone()
        t_end = t_start.clone()
        
        if classes is not None:
            t_start += self.label_emb(classes)
            t_mid += self.label_emb(classes)
            t_end += self.label_emb(classes)
        
        h = []
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(x, t_start)
            h.append(x)
            
            x = block2(x, t_start)
            x = attn(x)
            h.append(x)
            
            x = downsample(x)
        
        x = self.mid_block1(x, t_mid)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_mid)
        
        for i, (block1, block2, attn, upsample) in enumerate(self.ups):
            
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_mid)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_mid)
            x = attn(x)
            x = upsample(x)
        
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t_end)
        
        x = self.final_conv(x)
        
        return x
    
        
def Upsample(dim, dim_out=None, scale = 2):
    return nn.Sequential(
        nn.Upsample(scale_factor=(scale,2), mode='nearest'), nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

@torch.no_grad()
def validate(model, val_loader, utrlm_models=None, rnafm_models=None):
    """在验证集上评估模型，返回各项损失的详细信息"""
    model.eval()
    val_losses = []
    val_nll_losses = []
    val_dif_losses = []
    val_cls_losses = []
    val_utrlm_losses = []
    val_rnafm_losses = []
    
    for batch in tqdm(val_loader, desc='Validation'):
        x, y, sample_weight = batch
        x = x.type(torch.float32).to(device)
        y = y.type(torch.long).to(device)
        
        if args.only_ires:
            sample_weight = None
        else:
            sample_weight = sample_weight.to(device)
        
        # 随机时间步
        t = torch.randint(0, TIMESTEPS, (x.shape[0],)).long().to(device)
        
        # 计算损失
        nll_loss, dif_loss, cls_loss, utrlm_loss, rnafm_loss = p_losses(
            model, x, t, y, 
            loss_type="huber", 
            sample_weight=sample_weight, 
            class_weight=class_weight,
            utrlm_models=utrlm_models,
            rnafm_models=rnafm_models
        )
        
        # 记录各项损失
        val_nll_losses.append(nll_loss.item())
        val_dif_losses.append(dif_loss.item())
        val_cls_losses.append(cls_loss if isinstance(cls_loss, (int, float)) else cls_loss.item())
        val_utrlm_losses.append(utrlm_loss if isinstance(utrlm_loss, (int, float)) else utrlm_loss.item())
        val_rnafm_losses.append(rnafm_loss if isinstance(rnafm_loss, (int, float)) else rnafm_loss.item())
        
        if cls_loss != 0:
            loss = nll_loss + dif_loss + cls_loss
        else:
            loss = nll_loss + dif_loss
            
        val_losses.append(loss.item())
    
    # 返回各项损失的平均值
    return {
        'total_loss': np.mean(val_losses),
        'nll_loss': np.mean(val_nll_losses),
        'dif_loss': np.mean(val_dif_losses),
        'cls_loss': np.mean(val_cls_losses),
        'utrlm_loss': np.mean(val_utrlm_losses),
        'rnafm_loss': np.mean(val_rnafm_losses)
    }
    
# 预加载CLS模型（根据args.cls_model选择性加载）
utrlm_models, rnafm_models = None, None

print(f"Preloading CLS models for mode: {args.cls_model}")
utrlm_models, rnafm_models = load_all_cls_models(
    device, 
    args,  # 传入args而不是单独的文件路径
    utrlm_folds=utrlm_folds,
    rnafm_folds=rnafm_folds
)

# 打印加载结果
if utrlm_models:
    print(f"UTRLM models loaded: {len(utrlm_models)}")
if rnafm_models:
    print(f"RNAFM models loaded: {len(rnafm_models)}")
        
# model
model = Unet(
    dim=IMAGE_SIZE,
    channels=CHANNELS,
    dim_mults=(1, 2, 4),
    resnet_block_groups=RESNET_BLOCK_GROUPS,
    num_classes=TOTAL_CLASS_NUMBER,
).to(device)

ema = EMA(args.ema_beta)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

for name, param in model.named_parameters():
    if not torch.isfinite(param).all():
        print(f"Non-finite found in {name}")
        
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

live_kl = PlotLosses(groups={'DiffusionLoss': ['loss']}, outputs=[MatplotlibPlot()])

tf = T.Compose([T.ToTensor()])
# seq_dataset = SequenceDataset(sequences=train_data.Sequence, labels=train_data.IRES_class_600, class_weight = class_weight, transform=tf)
seq_dataset = SequenceDataset(
    sequences=train_data.Sequence.values,
    labels=train_data.IRES_class_600.values,
    class_weight=class_weight,
    transform=tf
)

val_dataset = SequenceDataset(
    sequences=val_data.Sequence.values,
    labels=val_data.IRES_class_600.values,
    class_weight=class_weight,
    transform=tf
)
train_dl = DataLoader(seq_dataset, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)  # , num_workers=3, pin_memory=True)

# 验证集数据加载器
# val_dataset = SequenceDataset(
#     sequences=val_data.Sequence, 
#     labels=val_data.IRES_class_600, 
#     class_weight=class_weight, 
#     transform=tf
# )
val_dl = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

gc.collect()
torch.cuda.empty_cache()

loss_list, nll_loss_list, dif_loss_list, cls_loss_list, utrlm_loss_list, rnafm_loss_list = [], [], [], [], [], []
val_losses = []
val_nll_losses = []
val_dif_losses = []
val_cls_losses = []
val_utrlm_losses = []
val_rnafm_losses = []
val_epochs = []  # 记录进行验证的epoch

patience = 20
epochs_without_improvement = 0
min_loss = np.inf
best_val_loss = np.inf
best_ep = 0

lr_history = []

# 在训练循环中添加这个验证
def verify_gradient_paths(model, nll_loss, dif_loss, cls_loss):
    """验证各个loss对主模型参数的梯度影响"""
    
    # 测试1: NLL + DIF loss的梯度
    model.zero_grad()
    (nll_loss + dif_loss).backward(retain_graph=True)
    main_grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
    
    # 测试2: CLS loss的梯度  
    model.zero_grad()
    if isinstance(cls_loss, torch.Tensor) and cls_loss.requires_grad:
        cls_loss.backward(retain_graph=True)
        cls_grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
    else:
        cls_grad_norm = 0.0
    
    print(f"主要损失梯度范数: {main_grad_norm:.8f}")
    print(f"CLS损失梯度范数: {cls_grad_norm:.8f}")
    
    # 测试3: 检查cls_loss的计算图
    if isinstance(cls_loss, torch.Tensor):
        print(f"cls_loss.grad_fn: {cls_loss.grad_fn}")
        
        # 追踪计算图
        current = cls_loss.grad_fn
        depth = 0
        while current is not None and depth < 10:
            print(f"  Level {depth}: {current}")
            if hasattr(current, 'next_functions'):
                current = current.next_functions[0][0] if current.next_functions else None
            else:
                break
            depth += 1
    
    return main_grad_norm, cls_grad_norm
for epoch in range(EPOCHS):
    epoch_losses, nll_epoch_losses, dif_epoch_losses, cls_epoch_losses, utrlm_epoch_losses, rnafm_epoch_losses = [], [], [], [], [], []
    batch_t = []
    model.train()

    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    
    # 训练一个epoch
    for step, batch in tqdm(enumerate(train_dl), desc=f'Epoch {epoch} Training'):
        x, y, sample_weight = batch
        x = x.type(torch.float32).to(device)
        y = y.type(torch.long).to(device)
        
        if args.only_ires:
            sample_weight = None
        else:
            sample_weight = sample_weight.to(device)
        
        t = torch.randint(0, TIMESTEPS, (x.shape[0],)).long().to(device)
        
        # Time warping逻辑
        if TIME_WARPING and len(dif_epoch_losses) >= 5 and step >= N_STEPS:
            sort_val = np.argsort(dif_epoch_losses[-len(batch_t):])
            sorted_t = [batch_t[i] for i in sort_val if i < len(batch_t)]
            
            if len(sorted_t) >= 5:
                last_n_t = sorted_t[-5:]
                unnested_last_n_t = [item for sublist in last_n_t for item in sublist]
                t_not_random = torch.tensor(np.random.choice(unnested_last_n_t, size=x.shape[0]), device=device)
                t = t if torch.rand(1) < 0.5 else t_not_random
        
        batch_t.append(list(t.cpu().detach().numpy()))

        # 计算损失
        nll_loss, dif_loss, cls_loss, utrlm_loss, rnafm_loss = p_losses(
            model, x, t, y, 
            loss_type="huber", 
            sample_weight=sample_weight, 
            class_weight=class_weight,
            utrlm_models=utrlm_models,  
            rnafm_models=rnafm_models
        )
        print(nll_loss, dif_loss, cls_loss, utrlm_loss, rnafm_loss)
        print(f"rnafm_loss requires_grad: {rnafm_loss.requires_grad}")
        print(f"rnafm_loss grad_fn: {rnafm_loss.grad_fn}")
        if rnafm_loss.grad_fn is not None:
            print("✓ rnafm_loss 可以反向传播")
        else:
            print("✗ rnafm_loss 无法反向传播")
            
        print(f"utrlm_loss requires_grad: {utrlm_loss.requires_grad}")
        print(f"utrlm_loss grad_fn: {utrlm_loss.grad_fn}")
        if utrlm_loss.grad_fn is not None:
            print("✓ utrlm_loss 可以反向传播")
        else:
            print("✗ utrlm_loss 无法反向传播")
            
        print(f"nll_loss requires_grad: {nll_loss.requires_grad}")
        print(f"nll_loss grad_fn: {nll_loss.grad_fn}")
        if nll_loss.grad_fn is not None:
            print("✓ nll_loss 可以反向传播")
        else:
            print("✗ nll_loss 无法反向传播")
            
        print(f"dif_loss requires_grad: {dif_loss.requires_grad}")
        print(f"dif_loss grad_fn: {dif_loss.grad_fn}")
        if dif_loss.grad_fn is not None:
            print("✓ dif_loss 可以反向传播")
        else:
            print("✗ dif_loss 无法反向传播")
        main_grad, cls_grad = verify_gradient_paths(model, nll_loss, dif_loss, cls_loss)
        # 记录损失
        nll_epoch_losses.append(nll_loss.mean().item())
        dif_epoch_losses.append(dif_loss.mean().item())
        cls_epoch_losses.append(cls_loss if isinstance(cls_loss, (int, float)) else cls_loss.item())
        utrlm_epoch_losses.append(utrlm_loss if isinstance(utrlm_loss, (int, float)) else utrlm_loss.item())
        rnafm_epoch_losses.append(rnafm_loss if isinstance(rnafm_loss, (int, float)) else rnafm_loss.item())
        
        # 总损失
        if cls_loss != 0: 
            loss = nll_loss + dif_loss + 100*cls_loss
        else:
            loss = nll_loss + dif_loss
        epoch_losses.append(loss.item())

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        ema.step_ema(ema_model, model)

    # 在epoch结束后进行验证（不是在训练循环内）
    if epoch % 5 == 0:  # 每5个epoch验证一次
        val_results = validate(model, val_dl, utrlm_models, rnafm_models)
        
        # 记录各项验证损失
        val_losses.append(val_results['total_loss'])
        val_nll_losses.append(val_results['nll_loss'])
        val_dif_losses.append(val_results['dif_loss'])
        val_cls_losses.append(val_results['cls_loss'])
        val_utrlm_losses.append(val_results['utrlm_loss'])
        val_rnafm_losses.append(val_results['rnafm_loss'])
        val_epochs.append(epoch)
        
        print(f"\nEpoch {epoch} - Train Loss: {np.mean(epoch_losses):.4f}, Val Loss: {val_results['total_loss']:.4f}, LR: {current_lr:.2e}")
        print(f"  Val NLL: {val_results['nll_loss']:.4f}, Val Dif: {val_results['dif_loss']:.4f}, Val CLS: {val_results['cls_loss']:.4f}")
        
        # 基于验证损失保存最佳模型
        if val_results['total_loss'] < best_val_loss:
            best_val_loss = val_results['total_loss']
            best_ep = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"/large_storage/goodarzilab/yanyichu/IRES-DM/models/{prefix}_best_model.pt")
            print(f"New best model saved at epoch {epoch}")
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement > patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # 记录训练损失
    loss_list.extend(epoch_losses)
    nll_loss_list.extend(nll_epoch_losses)
    dif_loss_list.extend(dif_epoch_losses)
    cls_loss_list.extend(cls_epoch_losses)
    utrlm_loss_list.extend(utrlm_epoch_losses)
    rnafm_loss_list.extend(rnafm_epoch_losses)

    # scheduler.step()
    
    # 打印训练进度
    if epoch % EPOCHS_LOSS_SHOW == 0:
        print(f"Epoch{epoch} | LR={current_lr:.2e} | Loss ｜ NLL={np.mean(nll_epoch_losses):.4f}|DIF={np.mean(dif_epoch_losses):.4f}|CLS={np.mean(cls_epoch_losses):.4f}|UTRLM={np.mean(utrlm_epoch_losses):.4f}|RNAFM={np.mean(rnafm_epoch_losses):.4f}")


print(model.state_dict()['final_res_block.res_conv.bias'])


# 在训练循环中使用


# 预期结果：
# - main_grad > 0 (正常)
# - cls_grad = 0 (说明cls_loss无法影响主模型参数)

# 修改绘图部分
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(12, 21))  # 图片尺寸调整得更合理
params = {
    'legend.fontsize': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
}
plt.rcParams.update(params)

# 计算每个epoch的平均训练损失
samples_per_epoch = len(train_dl)
epoch_avg_losses = {
    'nll': [],
    'dif': [],
    'cls': [],
    'utrlm': [],
    'rnafm': [],
    'total': []
}

# 创建batch到epoch的映射，用于x轴显示
batch_to_epoch = []
for i in range(epoch + 1):
    batch_to_epoch.extend([i] * samples_per_epoch)

# 计算每个epoch的平均值
for i in range(epoch + 1):
    start_idx = i * samples_per_epoch
    end_idx = min((i + 1) * samples_per_epoch, len(loss_list))
    if start_idx < len(loss_list) and end_idx > start_idx:
        epoch_avg_losses['nll'].append(np.mean(nll_loss_list[start_idx:end_idx]))
        epoch_avg_losses['dif'].append(np.mean(dif_loss_list[start_idx:end_idx]))
        epoch_avg_losses['cls'].append(np.mean(cls_loss_list[start_idx:end_idx]))
        epoch_avg_losses['utrlm'].append(np.mean(utrlm_loss_list[start_idx:end_idx]))
        epoch_avg_losses['rnafm'].append(np.mean(rnafm_loss_list[start_idx:end_idx]))
        epoch_avg_losses['total'].append(np.mean(loss_list[start_idx:end_idx]))

# 定义绘图函数以减少重复代码
def plot_loss(ax, loss_list, epoch_avg_loss, val_epochs, val_losses, title, ylabel='Loss'):
    # 绘制每个batch的损失（使用epoch作为x轴）
    if len(loss_list) > 0:
        x_batch = np.array(batch_to_epoch[:len(loss_list)])
        ax.plot(x_batch, loss_list, 'b-', alpha=0.2, linewidth=0.5, label='Train (per batch)')
    
    # 绘制每个epoch的平均损失
    if len(epoch_avg_loss) > 0:
        x_epoch = np.arange(len(epoch_avg_loss))
        ax.plot(x_epoch, epoch_avg_loss, 'b-', alpha=0.8, linewidth=2, label='Train (per epoch)')
    
    # 绘制验证损失
    if len(val_epochs) > 0 and len(val_losses) > 0:
        ax.plot(val_epochs, val_losses, 'ro-', markersize=6, linewidth=2, label='Validation')
    
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 设置x轴范围
    if epoch > 0:
        ax.set_xlim(-0.5, epoch + 0.5)

# 绘制各个损失
plot_loss(axes[0], nll_loss_list, epoch_avg_losses['nll'], 
          val_epochs, val_nll_losses, f'NLL Loss (Epoch {epoch})')

plot_loss(axes[1], dif_loss_list, epoch_avg_losses['dif'], 
          val_epochs, val_dif_losses, f'Diffusion Loss (Epoch {epoch})')

plot_loss(axes[2], cls_loss_list, epoch_avg_losses['cls'], 
          val_epochs, val_cls_losses, f'Classification Loss (Epoch {epoch})')

plot_loss(axes[3], utrlm_loss_list, epoch_avg_losses['utrlm'], 
          val_epochs, val_utrlm_losses, f'UTRLM Loss (Epoch {epoch})')

plot_loss(axes[4], rnafm_loss_list, epoch_avg_losses['rnafm'], 
          val_epochs, val_rnafm_losses, f'RNAFM Loss (Epoch {epoch})')

plot_loss(axes[5], loss_list, epoch_avg_losses['total'], 
          val_epochs, val_losses, f'Total Loss (Epoch {epoch})')

if len(lr_history) > 0:
    axes[6].plot(range(len(lr_history)), lr_history, 'g-', linewidth=2, label='Learning Rate')
    axes[6].set_title(f'Learning Rate Schedule (Epoch {epoch})')
    axes[6].set_xlabel('Epoch')
    axes[6].set_ylabel('Learning Rate')
    axes[6].set_yscale('log')  # 使用对数刻度
    axes[6].legend(loc='best')
    axes[6].grid(True, alpha=0.3)
    if epoch > 0:
        axes[6].set_xlim(-0.5, epoch + 0.5)
        
# 调整布局
plt.tight_layout()
plt.savefig(f'../results/{prefix}_TrainingPlot.png', dpi=150, bbox_inches='tight')
plt.close()

# 可选：保存损失数据到CSV以便后续分析
loss_data = pd.DataFrame({
    'epoch': range(len(epoch_avg_losses['total'])),
    'train_total': epoch_avg_losses['total'],
    'train_nll': epoch_avg_losses['nll'],
    'train_dif': epoch_avg_losses['dif'],
    'train_cls': epoch_avg_losses['cls'],
    'train_utrlm': epoch_avg_losses['utrlm'],
    'train_rnafm': epoch_avg_losses['rnafm']
})

# 添加验证数据
if len(val_epochs) > 0:
    val_df = pd.DataFrame({
        'epoch': val_epochs,
        'val_total': val_losses,
        'val_nll': val_nll_losses,
        'val_dif': val_dif_losses,
        'val_cls': val_cls_losses,
        'val_utrlm': val_utrlm_losses,
        'val_rnafm': val_rnafm_losses
    })
    loss_data = loss_data.merge(val_df, on='epoch', how='outer')

loss_data.to_csv(f'../results/{prefix}_training_history.csv', index=False)

model_state = torch.load(f"/large_storage/goodarzilab/yanyichu/IRES-DM/models/{prefix}_best_model.pt")
model.load_state_dict(model_state, strict = True)

final_val_results = validate(model, val_dl, utrlm_models, rnafm_models)
print(f"\nFinal Validation Results:")
print(f"  Total Loss: {final_val_results['total_loss']:.4f}")
print(f"  NLL Loss: {final_val_results['nll_loss']:.4f}")
print(f"  Diffusion Loss: {final_val_results['dif_loss']:.4f}")
print(f"  Class Loss: {final_val_results['cls_loss']:.4f}")
print(f"  UTRLM Loss: {final_val_results['utrlm_loss']:.4f}")
print(f"  RNAFM Loss: {final_val_results['rnafm_loss']:.4f}")

gen_seqs, real_seqs, random_seqs = process_and_evaluate_sequences(data, model, class_ = 1, seqlogo_max_len = IMAGE_SIZE,
                    kmer=False, hamming=False, gc=False, mfe=False, ss_hamming = False,
                    utrlm_models=utrlm_models,  
                    rnafm_models=rnafm_models ) 

if not args.only_ires:
    gen_seqs, real_seqs, random_seqs = process_and_evaluate_sequences(data, model, class_ = 0, seqlogo_max_len = IMAGE_SIZE, 
                        kmer=True, hamming=True, gc=True, mfe=True, ss_hamming = True,
                            utrlm_models=utrlm_models,  
                            rnafm_models=rnafm_models ) 
