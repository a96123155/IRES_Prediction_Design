import argparse
import copy
import gc
import math
import os
import random
from functools import partial
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import RNA
from collections import OrderedDict

import UTRLM_Predictor as utrlm
import RNAFM_Predictor as rnafm

import warnings
warnings.filterwarnings('ignore')

# 解析命令行参数
parser = argparse.ArgumentParser(description="IRES sequence generation using pre-trained model.")
parser.add_argument('--device', type=str, default='0', help='GPU device ID')
parser.add_argument('--model_path', type=str, 
                    default = '/large_storage/goodarzilab/yanyichu/IRES-DM/models/vMay23_IRESDM_174nt_wIRES_Both_Epoch300_L176_CutLeft_wIRES_Batch128_TimeSteps50_linearBETA_lr0.0001_CLSboth_best_model.pt',
                    help='Path to the pre-trained diffusion model')
parser.add_argument('--prefix', type=str, default='IRES_DM', help='Prefix for output files')
parser.add_argument('--image_size', type=int, default=176, help='Sequence length including padding')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for generation')
parser.add_argument('--timesteps', type=int, default=50, help='Timesteps for diffusion')
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to generate')
parser.add_argument('--class_id', type=int, default=1, help='Class ID to generate (0 or 1)')
parser.add_argument('--utrlm_modelfile', type=str, default='../models/IRES_UTRLM_best_model')
parser.add_argument('--rnafm_modelfile', type=str, default='../models/IRES_RNAFM_best_model')
parser.add_argument('--global_seed', type=int, default=42, help='Random seed')

args = parser.parse_args()

# 常量定义
NUCLEOTIDES = ['A', 'C', 'T', 'G']
CHANNELS = 1
RESNET_BLOCK_GROUPS = 4
IMAGE_SIZE = args.image_size
BATCH_SIZE = args.batch_size
TIMESTEPS = args.timesteps
N_SAMPLES = args.n_samples
GLOBAL_SEED = args.global_seed

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def seed_everything(seed=GLOBAL_SEED):
    """设置随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# 加载分类模型
def load_all_cls_models(device, args, utrlm_folds=10, rnafm_folds=10):
    """加载预训练的分类模型"""
    utrlm_models = []
    rnafm_models = []
    
    print("Loading UTRLM models...")
    for fold in range(utrlm_folds):
        try:
            state_dict = torch.load(f'{args.utrlm_modelfile}_fold{fold}.pt', map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.','')
                new_state_dict[name] = v
            
            model = utrlm.CNN_linear().to(device)
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            utrlm_models.append(model)
        except Exception as e:
            print(f"Failed to load UTRLM fold {fold}: {e}")
    
    print("Loading RNAFM models...")
    for fold in range(rnafm_folds):
        try:
            state_dict = torch.load(f'{args.rnafm_modelfile}_fold{fold}.pt', map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.','')
                new_state_dict[name] = v
            
            model = rnafm.RNAFM_linear().to(device)
            model.load_state_dict(new_state_dict, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            rnafm_models.append(model)
        except Exception as e:
            print(f"Failed to load RNAFM fold {fold}: {e}")
    
    print(f"Loaded {len(utrlm_models)} UTRLM models and {len(rnafm_models)} RNAFM models")
    return utrlm_models, rnafm_models

# Diffusion相关函数
def get_betas(timesteps=TIMESTEPS):
    """获取beta调度"""
    beta_start = 0.0001
    beta_end = 0.2
    return torch.linspace(beta_start, beta_end, timesteps)

# 初始化diffusion参数
betas = get_betas()
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def p_sample(model, x, t, t_index):
    """单步去噪采样"""
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    model_output = model(x, time=t, classes=None)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_guided(model, x, classes, t, t_index, context_mask, cond_weight=0.0):
    """带条件引导的采样"""
    batch_size = x.shape[0]
    t_double = t.repeat(2)
    x_double = x.repeat(2, 1, 1, 1)
    betas_t = extract(betas, t_double, x_double.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t_double, x_double.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t_double, x_double.shape)

    classes_masked = classes * context_mask
    classes_masked = classes_masked.type(torch.long)
    preds = model(x_double, time=t_double, classes=classes_masked)
    eps1 = (1 + cond_weight) * preds[:batch_size]
    eps2 = cond_weight * preds[batch_size:]
    x_t = eps1 - eps2

    model_mean = sqrt_recip_alphas_t[:batch_size] * (
        x - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, classes, shape, cond_weight):
    """完整的采样循环"""
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    
    if classes is not None:
        n_sample = classes.shape[0]
        context_mask = torch.ones_like(classes).to(device)
        classes = classes.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 0.0
        sampling_fn = partial(p_sample_guided, classes=classes, cond_weight=cond_weight, context_mask=context_mask)
    else:
        sampling_fn = partial(p_sample)

    for i in tqdm(reversed(range(0, TIMESTEPS)), desc='Sampling'):
        img = sampling_fn(model, x=img, t=torch.full((b,), i, device=device, dtype=torch.long), t_index=i)
        
    return img

@torch.no_grad()
def sample(model, image_size, classes=None, batch_size=16, channels=1, cond_weight=0):
    """生成样本"""
    result = p_sample_loop(model, classes=classes, shape=(batch_size, channels, len(NUCLEOTIDES), image_size), cond_weight=cond_weight)
    return result.cpu().numpy()

def sampling_to_sequences(model, number_of_samples=20, class_id=1, only_ires=False):
    """从模型采样生成序列"""
    seq_final = []
    total_count = 0
    
    for n_a in range(number_of_samples):
        if only_ires:
            random_classes = None
        else:
            sampled = torch.from_numpy(np.array([class_id] * BATCH_SIZE))
            random_classes = sampled.float().to(device)
        
        sampled_images = sample(
            model,
            classes=random_classes,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            channels=1,
            cond_weight=0,
        )
        
        for x in sampled_images:
            seq = ''.join([NUCLEOTIDES[s] for s in np.argmax(x.reshape(4, IMAGE_SIZE), axis=0)[1:-1]])
            total_count += 1
            
            if len(seq) == 174 and all(n in ['A', 'C', 'T', 'G'] for n in seq):
                seq_final.append(seq)
                    
    return list(set(seq_final)), total_count

def seq_to_fasta(seq_list, outfilename):
    """将序列保存为FASTA格式"""
    with open(outfilename, 'w') as file:
        for index, seq in enumerate(seq_list):
            file.write(f">seq_{index}_{len(seq)}\n{seq}\n")

def random_data(length=174, size=1000):
    """生成随机DNA序列"""
    samples = []
    for _ in range(size):
        random_seq = ''.join(random.choice(['A', 'T', 'C', 'G']) for _ in range(length))
        samples.append(random_seq)
    return samples

# U-Net模型定义（简化版，只包含必要部分）
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def l2norm(t):
    return F.normalize(t, dim=-1)

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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LearnedSinusoidalPosEmb(nn.Module):
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
        layers = [nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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

def Upsample(dim, dim_out=None, scale=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=(scale, 2), mode='nearest'), 
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

class Unet(nn.Module):
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

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.scale_factor = [2, 2]
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in, self.scale_factor[ind]) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ])
            )
            
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)

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

def evaluate_with_classifiers(sequences, utrlm_models, rnafm_models, class_id):
    """使用分类器评估生成的序列"""
    if len(sequences) == 0:
        return pd.DataFrame()
    
    # 初始化结果DataFrame
    results = pd.DataFrame()
    results['sequence'] = sequences
    
    # UTRLM评估
    if utrlm_models:
        utrlm_dataset, utrlm_dataloader = utrlm.generate_dataset_dataloader(
            [f'Generated_Class{class_id}'] * len(sequences), sequences
        )
        
        # 收集所有fold的预测结果
        utrlm_probs = []
        with torch.no_grad():
            for fold, model in enumerate(utrlm_models):
                utrlm_result = utrlm.predict_step(utrlm_dataloader, model, fold)
                utrlm_result = utrlm_result['df']
                # 只保留序列和概率列
                utrlm_probs.append(utrlm_result[f'Prob_U{fold}'].values)
        
        # 计算平均概率
        if utrlm_probs:
            utrlm_probs_array = np.array(utrlm_probs)
            results['Prob_U_Mean'] = np.mean(utrlm_probs_array, axis=0)
    
    # RNAFM评估
    if rnafm_models:
        rnafm_dataset, rnafm_dataloader = rnafm.generate_dataset_dataloader(
            [f'Generated_Class{class_id}'] * len(sequences), sequences
        )
        
        # 收集所有fold的预测结果
        rnafm_probs = []
        with torch.no_grad():
            for fold, model in enumerate(rnafm_models):
                rnafm_result = rnafm.predict_step(rnafm_dataloader, model, fold)
                rnafm_result = rnafm_result['df']
                # 只保留序列和概率列
                rnafm_probs.append(rnafm_result[f'Prob_R{fold}'].values)
        
        # 计算平均概率
        if rnafm_probs:
            rnafm_probs_array = np.array(rnafm_probs)
            results['Prob_R_Mean'] = np.mean(rnafm_probs_array, axis=0)
    
    # 计算总体平均概率
    if 'Prob_U_Mean' in results.columns and 'Prob_R_Mean' in results.columns:
        results['Prob_Mean'] = (results['Prob_U_Mean'] + results['Prob_R_Mean']) / 2
    elif 'Prob_U_Mean' in results.columns:
        results['Prob_Mean'] = results['Prob_U_Mean']
    elif 'Prob_R_Mean' in results.columns:
        results['Prob_Mean'] = results['Prob_R_Mean']
    
    # 添加序列长度
    results['length'] = [len(s) for s in results['sequence']]
    
    return results

def calculate_gc_content(sequence):
    """计算序列的GC含量"""
    if pd.isna(sequence) or len(sequence) == 0:
        return np.nan
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)

def calculate_mfe_and_ss(sequence):
    """使用ViennaRNA计算最小自由能和二级结构"""
    if pd.isna(sequence) or len(sequence) == 0:
        return np.nan, ""
    
    try:
        # 确保序列是大写的RNA序列
        sequence = sequence.upper().replace('T', 'U')
        
        # 计算MFE和二级结构
        ss, mfe = RNA.fold(sequence)
        return mfe, ss
    except:
        return np.nan, ""

def process_gc_mfe_ss(df):
    """处理单个dataframe，添加GC含量、MFE和SS列"""
    
    # 计算GC含量
    print("计算GC含量...")
    df['GC_content'] = df['sequence'].apply(calculate_gc_content)
    
    # 计算MFE和二级结构
    print("计算MFE和二级结构...")
    mfe_ss_results = []
    
    # 使用tqdm显示进度条
    for seq in tqdm(df['sequence'], desc=f"计算MFE/SS"):
        mfe, ss = calculate_mfe_and_ss(seq)
        mfe_ss_results.append((mfe, ss))
    
    # 将结果添加到dataframe
    df['MFE'] = [result[0] for result in mfe_ss_results]
    df['SS'] = [result[1] for result in mfe_ss_results]
    
    # 打印统计信息
    print(f"  GC含量: 均值={df['GC_content'].mean():.3f}, 标准差={df['GC_content'].std():.3f}")
    print(f"  MFE: 均值={df['MFE'].mean():.2f}, 标准差={df['MFE'].std():.2f}")
    print(f"  样本数: {len(df)}")
    
    return df
    
def main():
    # 创建输出目录
    os.makedirs('../results', exist_ok=True)
       
    # 加载分类模型
    print("Loading classification models...")
    utrlm_models, rnafm_models = load_all_cls_models(device, args)
    
    # 首先加载预训练权重来确定模型配置
    print("Loading diffusion model...")
    model_state = torch.load(args.model_path, map_location=device)
    
    # 从checkpoint中推断num_classes
    if 'label_emb.weight' in model_state:
        num_classes = model_state['label_emb.weight'].shape[0]
        print(f"Detected {num_classes} classes from checkpoint")
    else:
        # 如果没有label_emb，说明是无条件模型
        num_classes = None
        print("No label embedding found - using unconditional model")
    
    # 根据检测到的配置创建模型
    model = Unet(
        dim=IMAGE_SIZE,
        channels=CHANNELS,
        dim_mults=(1, 2, 4),
        resnet_block_groups=RESNET_BLOCK_GROUPS,
        num_classes=num_classes,
    ).to(device)
    
    # 加载预训练权重
    model.load_state_dict(model_state, strict=True)
    model.eval()
    
    # 更新only_ires标志基于模型配置
    only_ires = (num_classes == 1 or num_classes is None)
    if only_ires:
        print("Using IRES-only generation mode")
    else:
        print(f"Using conditional generation mode with {num_classes} classes")
    
    print("Model loaded successfully!")
    
    # 生成序列
    print(f"Generating sequences for class {args.class_id}...")
    number_of_batches = math.ceil(N_SAMPLES / BATCH_SIZE)
    gen_seqs, total_count = sampling_to_sequences(
        model, 
        number_of_samples=number_of_batches, 
        class_id=args.class_id,
        only_ires=only_ires
    )
    
    valid_count = len(gen_seqs)
    print(f'Generated: {valid_count}/{total_count} valid sequences')
    
    # 保存生成的序列
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # 使用分类器评估生成的序列
    if len(gen_seqs) > 0 and (utrlm_models or rnafm_models):
        print("Evaluating generated sequences with classifiers...")
        results = evaluate_with_classifiers(gen_seqs, utrlm_models, rnafm_models, args.class_id)
        
        
        if len(results) > 0:
            results_filename = f'../results/{args.prefix}_gen{N_SAMPLES}_Class{args.class_id}_{timestamp}.csv'
            results.to_csv(results_filename, index=False)
            print(f"Evaluation results saved to: {results_filename}")
            
            # 打印统计信息
            if 'Prob_Mean' in results.columns:
                print(f"Average IRES probability: {results['Prob_Mean'].mean():.4f}")
                print(f"High confidence sequences (>0.9): {(results['Prob_Mean'] > 0.9).sum()}")
        else:
            print("No evaluation results generated")
            
        results = process_gc_mfe_ss(results)
        results.to_csv(results_filename, index=False)
        
    else:
        print("Skipping classifier evaluation (no sequences or models)")
    
    print("Generation complete!")

if __name__ == "__main__":
    main()