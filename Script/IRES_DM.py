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
parser.add_argument('--prefix', type=str, default='IRESDM', help='Prefix for the run')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train for')
parser.add_argument('--image_size', type=int, default=200, help='Number of base pairs for the motif')
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

NUCLEOTIDES = ['A', 'C', 'T', 'G', 'N', 'E']
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

data = pd.read_csv('../Data/v2_dataset_with_unified_stratified_shuffle_train_test_split.csv', index_col = 0)
train_data = data[data.fold == 0]
if args.only_ires:
    TOTAL_CLASS_NUMBER = 1
    train_data = train_data[train_data.IRES_class_600 == 1]
    cell_types = [1]
    class_weight = None
else:
    TOTAL_CLASS_NUMBER = 2
    cell_types = [0, 1]
    class_weight = calculate_class_weight(torch.LongTensor(train_data.IRES_class_600.values))

if args.truncate_remain_right:
    train_data['Sequence']= [x[-(IMAGE_SIZE-1):]+'E' for x in train_data.Sequence]
else:
    train_data['Sequence']= [x[:(IMAGE_SIZE-1)]+'E' for x in train_data.Sequence]
    
train_data = train_data.reset_index(drop = True)#.iloc[:100]
train_data

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
    # print (x.shape, 'x_shape')
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, time=t, classes = None) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


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

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(model, classes, shape, cond_weight):
    device = next(model.parameters()).device
    b = shape[0]
#     print(b)
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
            seq = ''.join([NUCLEOTIDES[s] for s in np.argmax(x.reshape(len(NUCLEOTIDES), IMAGE_SIZE), axis=0)])
            total_count += 1
            
            if 'E' in seq:
                # 检查N是否在E之前出现
                if 'N' in seq and seq.index('N') < seq.index('E'):
                    continue  # 如果N在E之前出现，则跳过当前序列

                # 截断至第一个E出现的位置（EOS token）
                end_pos = seq.find('E')
                if end_pos != -1:
                    seq = seq[:end_pos]
                
            else:
                if 'N' in seq:
                    continue

            if seq != '': 
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

def calculate_weighted_nll_loss(x_start, pred_x_start, seq_mask=None, sample_weight=None):
    """
    使用样本权重和序列掩码计算加权的负对数似然损失。
    参数:
    - x_start: one-hot encoded targets [batch_size, 1, 5, seq_length]
    - pred_x_start: logits [batch_size, 1, 5, seq_length]
    - seq_mask: 有效序列位置的掩码 [batch_size, seq_length], 可选
    - sample_weight: 每个样本的权重 [batch_size], 可选
    """
    # 调整 pred_x_start 形状
    pred_x_start = pred_x_start.squeeze(1)#.permute(0, 2, 1)  # [batch_size, 5, seq_length]

    # x_start 转换为类别索引
    x_start = x_start.argmax(dim=2).squeeze(1)  # [batch_size, seq_length] [64, 200]

    # 处理掩码：将 ignore_index 应用于被掩码的位置
    if seq_mask is not None:
        # 设置忽略的索引为PAD N索引 -> -1
        x_start = x_start.masked_fill(~seq_mask.bool(), -1)
        
    # 定义交叉熵损失函数
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
    # 计算损失 
    nll_loss = loss_fct(pred_x_start, x_start)  # [batch_size, seq_length]
    
    # 应用样本权重
    if sample_weight is not None:
        # 广播样本权重到每个时间步
        sample_weight = sample_weight.unsqueeze(1).expand_as(nll_loss)
        nll_loss = nll_loss * sample_weight

    # 根据提供的 seq_mask 和 sample_weight 计算最终的平均损失
    if sample_weight is not None:
        total_loss = nll_loss.sum() / sample_weight.sum()
    else:
        total_loss = nll_loss.mean()

    return total_loss

def p_losses(denoise_model, x_start, t, classes, noise=None, seq_mask=None, sample_weight = None, loss_type="l1", p_uncond=0.1, class_weight = torch.Tensor([1., 1.])):
    """
    Calculate the loss conditioned and noise injected, ignoring the masked (padded) parts of the sequence.
    
    :param denoise_model: The denoising model to be evaluated.
    :param x_start: Tensor representing the original clean sequences.
    :param t: Time steps for the diffusion process.
    :param classes: Tensor of class labels for conditional guidance.
    :param noise: Optional noise tensor; if None, random noise will be generated.
    :param seq_mask: Tensor with the same shape as x_start, where 1 indicates data and 0 indicates padding.
    :param loss_type: Type of the loss function to apply ('l1', 'l2', or 'huber').
    :param p_uncond: Probability of using unconditional samples in training.
    
    :return: Computed loss value.
    """
    device = x_start.device
    if noise is None:
        noise = torch.randn_like(x_start)  # Generate Gaussian noise

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)  # Add noise based on t and generated noise

    # Create a mask for unconditional guidance
    if not args.only_ires:
        context_mask = torch.bernoulli(torch.zeros(classes.shape[0]) + (1 - p_uncond)).to(device)
        classes = classes * context_mask
        classes = classes.type(torch.long)
        predicted_noise = denoise_model(x_noisy, t, classes)
    else:
        predicted_noise = denoise_model(x_noisy, t, None)

    pred_x_start = predict_xstart_from_eps(x_noisy, t, predicted_noise)
    
    ##### NLL Loss
    nll_loss = calculate_weighted_nll_loss(x_start, pred_x_start, seq_mask, sample_weight)
    
    ##### Diffusion Loss
    # Apply sequence mask if provided
    if seq_mask is not None:
        seq_mask = seq_mask.unsqueeze(1).unsqueeze(2)
        seq_mask = seq_mask.expand(-1, 1, len(NUCLEOTIDES), -1)
        noise = noise * seq_mask
        predicted_noise = predicted_noise * seq_mask
    
    # Select the loss function
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
    dif_loss = dif_loss.sum() / seq_mask.sum() if seq_mask is not None else loss.mean()
    
    if args.cls_model:
        ##### CLS Loss
        pred_x_start_seqs = pred_x_start.squeeze(1).detach().cpu().numpy()

        pred_seqs, remain_idx = [], []
        for i, x in enumerate(pred_x_start_seqs):
            seq = ''.join([NUCLEOTIDES[s] for s in np.argmax(x, axis=0)])

            if 'E' in seq:
                # 检查N是否在E之前出现
                if 'N' in seq and seq.index('N') < seq.index('E'):
                    continue  # 如果N在E之前出现，则跳过当前序列

                # 截断至第一个E出现的位置（EOS token）
                end_pos = seq.find('E')
                if end_pos != -1:
                    seq = seq[:end_pos]

            else:
                if 'N' in seq:
                    continue

            if seq != '': 
                pred_seqs.append(seq)
                remain_idx.append(i)

        rnafm_loss, utrlm_loss = 0, 0
        if len(remain_idx) > 0:    
            classes = classes[remain_idx]
    #         print(len(classes))

            if args.cls_model != 'rnafm':
                utrlm_dataset, utrlm_dataloader = utrlm.generate_dataset_dataloader(classes.detach().cpu(), pred_seqs)

                utrlm_results = pd.DataFrame()
                for fold in range(utrlm_folds):
                    state_dict = torch.load(f'{args.utrlm_modelfile}_fold{fold}.pt', map_location=torch.device(device))
                    new_state_dict = OrderedDict()

                    for k, v in state_dict.items():
                        name = k.replace('module.','')
                        new_state_dict[name] = v

                    utrlm_model = utrlm.CNN_linear().to(device)
                    utrlm_model.load_state_dict(new_state_dict, strict = False)
                    utrlm_result = utrlm.predict_step(utrlm_dataloader, utrlm_model, fold)
                    utrlm_result.set_index('sequence', inplace=True)
                    if fold == 0:
                        utrlm_results = utrlm_result
                    else:
                        utrlm_results = pd.concat([utrlm_results, utrlm_result], axis=1)
                utrlm_prob_cols = [f'Prob_U{i}' for i in range(utrlm_folds)]
                utrlm_pred_cols = [f'Pred_U{i}' for i in range(utrlm_folds)]
                utrlm_results['Prob_U_Mean'] = utrlm_results[utrlm_prob_cols].mean(axis = 1)
                utrlm_results['Pred_U_Mean'] = utrlm_results[utrlm_pred_cols].mean(axis = 1)

                utrlm_prob = utrlm_results['Prob_U_Mean']
                utrlm_prob = np.stack([1 - utrlm_prob.values, utrlm_prob.values], axis = 1)
                utrlm_prob = torch.Tensor(utrlm_prob).to(device)
                if args.only_ires:
                    utrlm_loss = nn.MSELoss()(utrlm_prob[:,1], classes.float()).item()
                else:
                    utrlm_loss = nn.CrossEntropyLoss(weight = class_weight.to(device))(utrlm_prob, classes.long()).item()
            if args.cls_model != 'utrlm':
                rnafm_dataset, rnafm_dataloader = rnafm.generate_dataset_dataloader(classes.detach().cpu(), pred_seqs)

                rnafm_results = pd.DataFrame()
                for fold in range(rnafm_folds):
                    state_dict = torch.load(f'{args.rnafm_modelfile}_fold{fold}.pt', map_location=torch.device(device))
                    new_state_dict = OrderedDict()

                    for k, v in state_dict.items():
                        name = k.replace('module.','')
                        new_state_dict[name] = v

                    rnafm_model = rnafm.RNAFM_linear().to(device)
                    rnafm_model.load_state_dict(new_state_dict, strict = True)
                    rnafm_result = rnafm.predict_step(rnafm_dataloader, rnafm_model, fold)
                    rnafm_result.set_index('sequence', inplace=True)
                    if fold == 0:
                        rnafm_results = rnafm_result
                    else:
                        rnafm_results = pd.concat([rnafm_results, rnafm_result], axis=1)
                rnafm_prob_cols = [f'Prob_R{i}' for i in range(rnafm_folds)]
                rnafm_pred_cols = [f'Pred_R{i}' for i in range(rnafm_folds)]
                rnafm_results['Prob_R_Mean'] = rnafm_results[rnafm_prob_cols].mean(axis = 1)
                rnafm_results['Pred_R_Mean'] = rnafm_results[rnafm_pred_cols].mean(axis = 1)

                rnafm_prob = rnafm_results['Prob_R_Mean']
                rnafm_prob = np.stack([1 - rnafm_prob.values, rnafm_prob.values], axis = 1)
                rnafm_prob = torch.Tensor(rnafm_prob).to(device)
                if args.only_ires:
                    rnafm_loss = nn.MSELoss()(rnafm_prob[:,1], classes.float()).item()
                else:
                    rnafm_loss = nn.CrossEntropyLoss(weight = class_weight.to(device))(rnafm_prob, classes.long()).item()
            cls_loss = utrlm_loss + rnafm_loss
    else:
        cls_loss = 0
        rnafm_loss, utrlm_loss = 0, 0
    return nll_loss, dif_loss, cls_loss, utrlm_loss, rnafm_loss

class SequenceDataset(Dataset):
    """
    Characterizes a dataset for PyTorch, including one-hot encoding of nucleotide sequences.
    """
    def __init__(self, sequences, labels, class_weight = None, transform=None):
        """
        Initialization.
        :param sequences: List of nucleotide sequences.
        :param labels: Corresponding labels for the sequences.
        :param transform: Optional transform to be applied on a sample.
        """
        self.labels = labels
        self.nucleotides = NUCLEOTIDES
        self.max_seq_len = IMAGE_SIZE
        self.transform = transform
        self.padded_seqs = []  # Store padded sequences
        self.images = []
        self.seq_mask = []
        self.class_weight = class_weight

        for seq in sequences:
            padded_seq, encoded_img = self.one_hot_encode(seq)
            self.padded_seqs.append(padded_seq)  # Store the padded sequence
            self.images.append(encoded_img)
            self.seq_mask.append([0 if nucleotide == 'N' else 1 for nucleotide in padded_seq])

        self.images = np.array([x.T.tolist() for x in self.images])
        self.seq_mask = np.array(self.seq_mask)

    def one_hot_encode(self, seq):
        """
        One-hot encodes a nucleotide sequence, padding with 'N' up to max_seq_len.
        :param seq: A string of nucleotide sequence.
        :return: tuple of the padded sequence and one-hot encoded numpy array.
        """
        padded_seq = seq.ljust(self.max_seq_len, 'N')  # Pad sequence with 'N' on the right
        seq_array = np.zeros((self.max_seq_len, len(self.nucleotides)))
        for i, nucleotide in enumerate(padded_seq):
            if nucleotide in self.nucleotides:
                seq_array[i, self.nucleotides.index(nucleotide)] = 1
        return padded_seq, seq_array

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Generates one sample of data.
        """
        image = self.images[index]
        label = self.labels[index]
        seq_mask = self.seq_mask[index]
        padded_seq = self.padded_seqs[index]  # Retrieve the padded sequence
        if self.transform:
            image = self.transform(image)
        
        if self.class_weight is not None: 
            sample_weight = self.class_weight[label]
        else:
            sample_weight = torch.tensor([1.0])
        return image, label, sample_weight, torch.from_numpy(seq_mask).type(torch.float32), padded_seq

def random_data(length, size, focus_start=174, focus_end=200, focus_prob=0.8):
    """
    生成随机数据序列，其中 'E' 在 175-220 之间出现的概率更高。

    参数:
        length (int): 每个序列的长度
        size (int): 生成的序列数量
        focus_start (int): 'E' 主要出现的起始位置
        focus_end (int): 'E' 主要出现的结束位置
        focus_prob (float): 'E' 在主要区域内出现的概率
    返回:
        list: 生成的序列列表
    """
    samples = []
    for _ in range(size):
        # 首先生成不包含 'E' 的随机序列
        sample = ''.join(random.choice(['A', 'T', 'C', 'G']) for _ in range(length))
        
        # 决定 'E' 是否应该出现在主要集中区域
        if random.random() < focus_prob:
            # 主要集中区域放 'E'
            e_position = random.randint(focus_start, min(focus_end, length - 1))
        else:
            # 其他区域放 'E' 的概率较低
            e_position = random.randint(1, length - 1)
        
        # 将 'E' 插入到序列中
        sample = sample[:e_position]
        samples.append(sample)

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
            
    
def create_frequency_matrix(sequences, max_len = IMAGE_SIZE, truncate_remain_right=False):
    """从DNA序列列表创建频率矩阵，可以选择保留右侧多出的部分或者左侧填充"""
    # 假设所有序列长度相同，获取序列长度
    seq_length = max(len(s) for s in sequences)
    
    # 根据 truncate_remain_right 标志来填充 'N'
    if args.truncate_remain_right:
        sequences = ['N'*(seq_length - len(s)) + s if len(s) < seq_length else s for s in sequences]
    else:
        sequences = [s + 'N'*(seq_length - len(s)) if len(s) < seq_length else s for s in sequences]
    
    # 初始化计数矩阵
    count_matrix = pd.DataFrame(0, index=np.arange(seq_length), columns=['A', 'C', 'G', 'T'])
    
    # 计算每个位置上各核苷酸的出现次数
    for seq in sequences:
        for index, nucleotide in enumerate(seq):
            if nucleotide in count_matrix.columns:
                count_matrix.loc[index, nucleotide] += 1
    
    # 转换为频率矩阵
    frequency_matrix = count_matrix.div(len(sequences))
    
    # 如果 IMAGE_SIZE 大于序列长度，进行填充
    if max_len > seq_length:
        pad_width = max_len - seq_length  # 计算需要填充的行数
        # 使用 np.pad 进行填充，然后转换回 DataFrame
        padded_matrix = np.pad(frequency_matrix.values, ((pad_width, 0), (0, 0)), 'constant', constant_values=(0,))
        padded_matrix = pd.DataFrame(padded_matrix, columns=frequency_matrix.columns)
    else:
        padded_matrix = frequency_matrix.iloc[-max_len:]

    return padded_matrix

def process_and_evaluate_sequences(train_data, model_best, class_ = 1, seqlogo_max_len = IMAGE_SIZE, kmer=False, hamming=False, gc=True, mfe=False, ss_hamming = False):
    # Selecting real sequences based on class and length
    real_seqs = train_data[(train_data.IRES_class_600 == class_) & (train_data.Length == IMAGE_SIZE)]['Sequence'].values
    if len(real_seqs) < 1000:
        real_seqs = train_data[train_data.IRES_class_600 == class_]['Sequence'].values
    real_seqs = [s[:-1] for s in real_seqs]
    
    # Generating random sequences
    random_seqs = random_data(length=IMAGE_SIZE, size=len(real_seqs))
    
    # Model evaluation and sequence generation
    model_best.eval()
    number_of_batches = [1, int(len(real_seqs)/BATCH_SIZE)][int(len(real_seqs)/BATCH_SIZE) > 0]
    gen_seqs, total_count = sampling_to_metric(model_best, number_of_samples=number_of_batches,  
                                  specific_group=True, group_number=class_)
    
    valid_count = len(gen_seqs)
    print(f'MaxLength = {IMAGE_SIZE} | Num: Correct_Generate = {valid_count}/{total_count} | Real = {len(real_seqs)} | Random = {len(random_seqs)}')
    
    # Saving generated sequences to a FASTA file
    seq_to_fasta(gen_seqs, f'../results/{prefix}_Gene_Class{class_}_num{valid_count}_ep{best_ep}.fasta')
    seq_to_fasta(real_seqs, f'../results/{prefix}_Real_Class{class_}_num{len(real_seqs)}_ep{best_ep}.fasta')
    seq_to_fasta(random_seqs, f'../results/{prefix}_Rand_Class{class_}_num{len(random_seqs)}_ep{best_ep}.fasta')
    
    # UTR-LM
    utrlm_dataset, utrlm_dataloader = utrlm.generate_dataset_dataloader([f'Generated_Class{class_}']*len(gen_seqs), gen_seqs)

    utrlm_results = pd.DataFrame()
    for fold in range(utrlm_folds):
        state_dict = torch.load(f'{args.utrlm_modelfile}_fold{fold}.pt', map_location=torch.device(device))
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k.replace('module.','')
            new_state_dict[name] = v

        utrlm_model = utrlm.CNN_linear().to(device)
        utrlm_model.load_state_dict(new_state_dict, strict = False)
        utrlm_result = utrlm.predict_step(utrlm_dataloader, utrlm_model, fold)
        utrlm_result.set_index('sequence', inplace=True)
        if fold == 0:
            utrlm_results = utrlm_result
        else:
            utrlm_results = pd.concat([utrlm_results, utrlm_result], axis=1)
    utrlm_prob_cols = [f'Prob_U{i}' for i in range(utrlm_folds)]
    utrlm_pred_cols = [f'Pred_U{i}' for i in range(utrlm_folds)]
    utrlm_results['Prob_U_Mean'] = utrlm_results[utrlm_prob_cols].mean(axis = 1)
    utrlm_results['Pred_U_Mean'] = utrlm_results[utrlm_pred_cols].mean(axis = 1)
    # RNA-FM
    rnafm_dataset, rnafm_dataloader = rnafm.generate_dataset_dataloader([f'Generated_Class{class_}']*len(gen_seqs), gen_seqs)

    rnafm_results = pd.DataFrame()
    for fold in range(rnafm_folds):
        state_dict = torch.load(f'{args.rnafm_modelfile}_fold{fold}.pt', map_location=torch.device(device))
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k.replace('module.','')
            new_state_dict[name] = v

        rnafm_model = rnafm.RNAFM_linear().to(device)
        rnafm_model.load_state_dict(new_state_dict, strict = True)
        rnafm_result = rnafm.predict_step(rnafm_dataloader, rnafm_model, fold)
        rnafm_result.set_index('sequence', inplace=True)
        if fold == 0:
            rnafm_results = rnafm_result
        else:
            rnafm_results = pd.concat([rnafm_results, rnafm_result], axis=1)
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
        self.scale_factor = [3,2]
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
    
        
def Upsample(dim, dim_out=None, scale = 3):
    return nn.Sequential(
        nn.Upsample(scale_factor=(scale,2), mode='nearest'), nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

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

live_kl = PlotLosses(groups={'DiffusionLoss': ['loss']}, outputs=[MatplotlibPlot()])

tf = T.Compose([T.ToTensor()])
seq_dataset = SequenceDataset(sequences=train_data.Sequence, labels=train_data.IRES_class_600, class_weight = class_weight, transform=tf)
train_dl = DataLoader(seq_dataset, BATCH_SIZE, shuffle=True)  # , num_workers=3, pin_memory=True)

gc.collect()
torch.cuda.empty_cache()

loss_list, nll_loss_list, dif_loss_list, cls_loss_list, utrlm_loss_list, rnafm_loss_list = [], [], [], [], [], []
min_loss = np.inf
for epoch in range(EPOCHS):
    epoch_losses, nll_epoch_losses, dif_epoch_losses, cls_epoch_losses, utrlm_epoch_losses, rnafm_epoch_losses = [], [], [], [], [], []
    batch_t = []
    model.train()
    for step, batch in tqdm(enumerate(train_dl)):
        x, y, sample_weight, seq_mask, seq = batch
        x = x.type(torch.float32).to(device)
        y = y.type(torch.long).to(device)
        seq_mask = seq_mask.to(device)
        if args.only_ires:
            sample_weight = None
        else:
            sample_weight = sample_weight.to(device)
        
        t = torch.randint(0, TIMESTEPS, (x.shape[0],)).long().to(device)  # *np.random.uniform()
        
        if TIME_WARPING == True and step >= N_STEPS:
            # sort the epoch losses so that one can take the t-s with the biggest losses
            sort_val = np.argsort(dif_epoch_losses)
            sorted_t = [batch_t[i] for i in sort_val]
            
            # take the t-s for the 5 biggest losses (5 was taken as example, no extensive optimization)
            last_n_t = sorted_t[-5:]
            unnested_last_n_t = [item for sublist in last_n_t for item in sublist]
            
            # take x.shape[0] number of t-s for the 5 biggest losses
            t_not_random = torch.tensor(np.random.choice(unnested_last_n_t, size=x.shape[0]), device="cpu")
            # pick between t generated above and t_not_random (to increase exploration, and not to get stuck
            # in the same t-s)
            
            t= t if torch.rand(1) < 0.5 else t_not_random
            t = t.to(device)
        
        batch_t.append(list(t.cpu().detach().numpy()))

        nll_loss, dif_loss, cls_loss, utrlm_loss, rnafm_loss = p_losses(model, x, t, y, seq_mask = seq_mask, loss_type="huber", sample_weight=sample_weight, class_weight = class_weight)
        nll_epoch_losses.append(nll_loss.mean().item())
        dif_epoch_losses.append(dif_loss.mean().item())
        cls_epoch_losses.append(np.mean(cls_loss))
        utrlm_epoch_losses.append(np.mean(utrlm_loss))
        rnafm_epoch_losses.append(np.mean(rnafm_loss))
        
        if cls_loss != 0: 
            loss = nll_loss + dif_loss + cls_loss
        else:
            loss = nll_loss + dif_loss
        epoch_losses.append(loss.item())

        if loss.item() < min_loss:
            best_ep = epoch
            torch.save(model.state_dict(), f"../models/{prefix}_best_model.pt")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.step_ema(ema_model, model)
#         break
    if (epoch % EPOCHS_LOSS_SHOW) == 0:
        print(f"Epoch{epoch} | Loss ｜ NLL={sum(nll_epoch_losses):.4f}|DIF={sum(dif_epoch_losses):.4f}|CLS={sum(cls_epoch_losses):.4f}|UTRLM={sum(utrlm_epoch_losses):.4f}|RNAFM={sum(rnafm_epoch_losses):.4f}",)

    loss_list.extend(epoch_losses)
    nll_loss_list.extend(nll_epoch_losses)
    dif_loss_list.extend(dif_epoch_losses)
    cls_loss_list.extend(cls_epoch_losses)
    utrlm_loss_list.extend(utrlm_epoch_losses)
    rnafm_loss_list.extend(rnafm_epoch_losses)
    
    fig, axes = plt.subplots(nrows = 6, ncols = 1, figsize = (45,50))  # 设置图形的大小
    params = {
        'legend.fontsize': 20,  # 图表大小
        'axes.labelsize': 20,        # 轴标签字体大小
        'axes.titlesize': 20,        # 轴标题字体大小
        'xtick.labelsize': 20,       # X轴刻度标签字体大小
        'ytick.labelsize': 20        # Y轴刻度标签字体大小
    }
    plt.rcParams.update(params)  # 应用绘图参数
    axes[0].plot(list(range(len(nll_loss_list))), nll_loss_list, linestyle='-', color='b')  # 画线，可以选择不同的marker
    axes[0].set_title(f'NLL Loss | Epoch = {epoch}')
    axes[1].plot(list(range(len(dif_loss_list))), dif_loss_list, linestyle='-', color='b')  # 画线，可以选择不同的marker
    axes[1].set_title(f'Diffusion Loss | Epoch = {epoch}')
    axes[2].plot(list(range(len(cls_loss_list))), cls_loss_list, linestyle='-', color='b')  # 画线，可以选择不同的marker
    axes[2].set_title(f'Class Belonging Loss | Epoch = {epoch}')
    axes[3].plot(list(range(len(utrlm_loss_list))), utrlm_loss_list, linestyle='-', color='b')  # 画线，可以选择不同的marker
    axes[3].set_title(f'UTRLM Class Belonging Loss | Epoch = {epoch}')
    axes[4].plot(list(range(len(rnafm_loss_list))), rnafm_loss_list, linestyle='-', color='b')  # 画线，可以选择不同的marker
    axes[4].set_title(f'RNAFM Class Belonging Loss | Epoch = {epoch}')
    axes[5].plot(list(range(len(loss_list))), loss_list, linestyle='-', color='b')  # 画线，可以选择不同的marker
    axes[5].set_title(f'Loss | Epoch = {epoch}')
   
    axes[0].set_ylabel('Loss')  # Y轴标签
    axes[0].set_xlabel('Epochs')  # X轴标签
    axes[1].set_xlabel('Epochs')  # X轴标签
    axes[2].set_xlabel('Epochs')  # X轴标签
    axes[3].set_xlabel('Epochs')  # X轴标签
    axes[4].set_xlabel('Epochs')  # X轴标签
    axes[5].set_xlabel('Epochs')  # X轴标签
    
    axes[0].grid(True)  # 显示网格
    axes[1].grid(True)  # 显示网格
    axes[2].grid(True)  # 显示网格
    axes[3].grid(True)  # 显示网格
    axes[4].grid(True)  # 显示网格
    axes[5].grid(True)  # 显示网格
    plt.savefig(f'../results/{prefix}_TrainingPlot.png')

model_state = torch.load(f"../models/{prefix}_best_model.pt")
model.load_state_dict(model_state, strict = True)
gen_seqs, real_seqs, random_seqs = process_and_evaluate_sequences(train_data, model, class_ = 1, seqlogo_max_len = IMAGE_SIZE,
                    kmer=True, hamming=True, gc=True, mfe=True, ss_hamming = True)

if not args.only_ires:
    gen_seqs, real_seqs, random_seqs = process_and_evaluate_sequences(train_data, model, class_ = 0, seqlogo_max_len = IMAGE_SIZE, 
                        kmer=True, hamming=True, gc=True, mfe=True, ss_hamming = True)

