# Comprehensive IRES Identification, Directed Mutation, and De Novo Generation Using AI-Driven Methods

Internal ribosome entry sites (IRES) within RNA enable ribosome recruitment and translation initiation, especially during cap-independent translation initiation. IRES are potential therapeutic targets for regulating viral protein expression. Our study presents an innovative framework for IRES research. IRES-LM, a language model transferred from UTR-LM and RNA-FM, is trained on 46,774 sequences to predict linear mRNA IRESs, outperforming existing methods with a 15% improvement in AUC and F1 score. Additionally, IRES-LM demonstrates remarkable cross-applicability to circRNA IRESs, correctly identifying all experimentally validated 21 IRESs and outperforming benchmark methods. Next, IRES-EA combines IRES-LM with an evolutionary algorithm for directed mutation. We tested 37,293 non-IRES sequences, successfully mutating 22,243 into IRES sequences. Moreover, experimental validation shows that 1 out of 7 model-generated EMCV IRES variants achieved higher activity than wild-type EMCV IRES, where secondary structure comparison identifies potential key elements for future engineering. For CVB3 IRES variants, 1 out of 10 model-generated mutants achieved comparable activity to the wild-type, with all mutated sequences exhibiting detectable activity above the negative control. Further, IRES-DM employs a conditional diffusion model to de novo generate novel IRES sequences, resembling natural IRES in structure and function, exemplified by a generated sequence sharing only 27.6% sequence similarity with BiP IRES while maintaining similar secondary structure. Experimental testing of two generated sequences similar to VCIP IRES demonstrated substantial IRES activity despite being only 95-nt in length. This comprehensive framework advances IRES identification, optimization, and design, highlighting its potential for enhancing RNA vaccine efficacy and broader therapeutic applications.

## Overview

This framework consists of three main components:

1. **IRES Language Model for IRES Identification (IRES-LM)**
   - **IRES-UTRLM**: A language model fine-tuned on untranslated regions (UTRs)
   - **IRES-RNAFM**: A language model based on non-coding RNA sequences

2. **IRES Evolutionary Algorithm (IRES-EA)**
   - Enhancing IRES activity through guided mutations
   - Supporting multiple mutation strategies

3. **IRES Diffusion Model (IRES-DM)**
   - De novo generation of functional IRES sequences
   - Two training strategies: reward-guided and directed training

## Dataset

- `./Data/v2_dataset_with_unified_stratified_shuffle_train_test_split.csv`: Contains 9,072 IRES and 37,602 non-IRES sequences
- `./Data/circires_case.fa`: 21 experimentally validated circRNA IRESs collected from DeepCIP

## Installation

Typical install time on a "normal" desktop computer < 1 hour
```bash
git clone https://github.com/a96123155/IRES_Prediction_Design.git
cd IRES_Prediction_Design
conda create -n ires python==3.9 # or 3.12
pip install requirements.txt
```

## 1. IRES Language Model (IRES-LM)

IRES-LM combines two fine-tuned language models: IRES-UTRLM (based on untranslated regions) and IRES-RNAFM (based on non-coding RNA sequences).

### 1.1 IRES-RNAFM

#### Command
```bash
cd ./Script
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 8890 IRES_RNAFM.py --device_ids 0 --bos_emb --truncate --finetune_esm --prefix IRES_RNAFM
```

#### Parameters
```
Namespace(device_ids='0', local_rank=0, seed=1337, prefix='IRES_RNAFM', epochs=10, nodes=40, dropout3=0.5, lr=0.0001, folds=10, random_init=False, avg_emb=False, bos_emb=True, epoch_without_improvement=10, pos_label_weight=-1, cls_loss_weight=2.0, mlm_loss_weight=1, finetune_esm=True, finetune_sixth_layer_esm=False, truncate=True, truncate_num=1024, mask_prob=0.15, batch_toks=4096)
```

#### Performance Metrics
- `./results/IRES_RNAFM_CLS2_10folds_AvgEmbFalse_BosEmbTrue_epoch10_nodes40_dropout30.5_finetuneESMTrue_finetuneLastLayerESMFalse_lr0.0001_metrics.csv`

#### Saved Models
- `./models/IRES_RNAFM_best_model_fold[0-9].pt`
please find in [link](https://drive.google.com/drive/folders/13U1VLNbTH75nqp_ieRUr6uOAp3IrJ8RU?usp=drive_link)

### 1.2 IRES-UTRLM

#### Pretrained Model
- `../models/UTRLM_pretrained_model.pkl`
please find in [link](https://drive.google.com/drive/folders/13U1VLNbTH75nqp_ieRUr6uOAp3IrJ8RU?usp=drive_link)

#### Command
```bash
cd ./Script
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 8889 IRES_UTRLM.py --device_ids 0 --bos_emb --finetune_esm --prefix IRES_UTRLM
```

#### Parameters
```
Namespace(device_ids='0', local_rank=3, seed=1337, prefix='IRES_UTRLM', epochs=100, nodes=40, dropout3=0.5, lr=0.0001, random_init=False, modelfile='UTRLM_pretrained_model.pkl', avg_emb=False, bos_emb=True, epoch_without_improvement=10, pos_label_weight=-1, cls_loss_weight=20.0, mlm_loss_weight=1, finetune_esm=True, finetune_sixth_layer_esm=False, truncate=False, truncate_num=50, mask_prob=0.15, batch_toks=2048)
```

#### Performance Metrics
- `./results/IRES_UTRLM_FinetuneESM_lr1e-4_dr5_bos_CLS20_10folds_F0_AvgEmbFalse_BosEmbTrue_epoch100_nodes40_dropout30.5_finetuneESMTrue_finetuneLastLayerESMFalse_lr0.0001_metrics.csv`

#### Saved Models
- `./models/IRES_UTRLM_best_model_fold[0-9].pt`
please find in [link](https://drive.google.com/drive/folders/13U1VLNbTH75nqp_ieRUr6uOAp3IrJ8RU?usp=drive_link)

### 1.3 Combined IRES-LM Results

- `./Results/IRESLM_experimentally_validated_circRNA_IRES_n21.csv`: Performance comparison in predicting 21 experimentally validated circRNA IRESs. IRES-LM, which averages predictions from IRES-UTRLM and IRES-RNAFM, identified all 21 IRESs. In this file, "U" represents IRES-UTRLM, "R" represents IRES-RNAFM.
- `./IRESLM_UMAP_Representation_MFE`: UMAP visualization of IRES-LM representations colored by minimum free energy (MFE) values.

### 1.4 Running Time

Training was conducted on NVIDIA Tesla GPUs with 32GB memory per GPU, with IRES-UTRLM requiring approximately 355 seconds per epoch across 4 GPUs, while IRES-RNAFM required approximately 828 seconds per epoch. 

## 2. IRES Evolutionary Algorithm (IRES-EA)

IRES-EA combines IRES-LM with an evolutionary algorithm to enhance IRES activity through guided mutations.

### IRES-EA Parameters Explanation

#### Basic Configuration
- `--prefix`: Sets the output file prefix (default: 'VCIP_gene_editing_AtoG')
- `--wt_seq`: Specifies the wild-type sequence to be mutated (default: VCIP sequence)
- `--nb_folds`: Number of cross-validation models to use (default: 10)
- `--model_filename`: Path to the model file (default: 'IRES_RNAFM_CLS2_best_model')

#### Sequence Mutation Control
- `--start_nt_position`: Starting position for mutations (0-based indexing)
- `--end_nt_position`: Ending position for mutations (-1 means the last position)
- `--n_mut`: Maximum number of mutations allowed for each sequence (default: 3)
- `--mutation_type`: Mutation strategy to use:
  - `mut_random`: Random mutations
  - `mut_attn`: Attention-guided mutations
  - `mut_gene_editing`: Base-specific editing (e.g., A→G)
- `--nt_replacements`: For gene editing mode, defines nucleotide substitutions (format: "A=G,C=T")
- `--mutate2stronger`: When enabled, directs mutations toward stronger IRES activity

#### Algorithm Parameters
- `--mut_by_prob`: When enabled, uses probability-based mutation instead of transformed probability based on `transform_type`
- `--transform_type`: Probability transformation method:
  - `logit`: Logistic transformation
  - `sigmoid`: Sigmoid function
  - `power_law`: Power law transformation
  - `tanh`: Hyperbolic tangent
- `--mlm_tok_num`: Number of masked tokens per sequence in each iteration (default: 1)
- `--n_designs_ep`: Number of candidate mutations generated per iteration (default: 10)
- `--n_sampling_designs_ep`: Number of mutations sampled from candidates (default: 5)
- `--n_mlm_recovery_sampling`: Number of sampling attempts for masked language model recovery (default: 1)


### 2.1 IRES-UTRLM Based EA

#### Command
```bash
CUDA_VISIBLE_DEVICES=0 python3 IRES_UTRLM_EA.py --prefix VCIP --wt_seq CAGCCTCGGCCAGGAGGCGACCCGGGCGCCTGGGTGTGTGGCTGCTGTTGCGGGACGTCTTCGCGGGGCGGGAGGCTCGCGCCGCAGCCAGCGCC --n_mut 10 --mutate2stronger --mutation_type mut_random
```

### 2.2 IRES-RNAFM Based EA

#### Command
```bash
CUDA_VISIBLE_DEVICES=0 python3 IRES_RNAFM_EA.py --prefix VCIP --wt_seq CAGCCTCGGCCAGGAGGCGACCCGGGCGCCTGGGTGTGTGGCTGCTGTTGCGGGACGTCTTCGCGGGGCGGGAGGCTCGCGCCGCAGCCAGCGCC --n_mut 10 --mutate2stronger --mutation_type mut_random
```

### 2.3 Prediction on Mutated Sequences

#### Command
```bash
cd ./Script
CUDA_VISIBLE_DEVICES=0 python3 IRESLM_Predict_IRESEA_mutations.py --data_file ../IRES_EA/vJul3_UTRLM_Design_3619_random_Mut10Sites_StrongerTrue_MutByProbFalse.csv --column_name MT --prefix vJul12_vJul3_UTRLM_Design_3619_random_Mut10Sites_StrongerTrue_MutByProbFalse
```

## 3. IRES Diffusion Model (IRES-DM)

IRES-DM employs a conditional diffusion model for de novo generation of functional IRES sequences.

### 3.1 Reward-guided Training

#### Command
```bash
cd ./Script
CUDA_VISIBLE_DEVICES=0 python3 IRES_DM.py --only_ires --prefix IRESDM_RewardGuided --epochs 150 --cls_model utrlm
```

#### Parameters
```
Namespace(device='0', prefix='IRESDM_RewardGuided', epochs=150, image_size=200, only_ires=True, batch_size=64, timesteps=50, beta_scheduler='linear', learning_rate=0.0001, global_seed=42, ema_beta=0.995, n_steps=10, n_samples=1000, save_and_sample_every=1, epochs_loss_show=5, time_warping=True, truncate_remain_right=True, utrlm_modelfile='../models/IRES_UTRLM_best_model', rnafm_modelfile='../models/IRES_RNAFM_best_model', cls_model='utrlm')
```

#### Generated Sequences
- `./results/IRESDM_RewardGuided_CLSutrlm_Gene_Class1_correctNum6178_ep149.csv`

#### Saved Model
- `./models/IRESDM_RewardGuided_CLSutrlm_best_model.pt`
please find in [link](https://drive.google.com/drive/folders/13U1VLNbTH75nqp_ieRUr6uOAp3IrJ8RU?usp=drive_link)

### 3.2 Directed Training

#### Command
```bash
cd ./Script
CUDA_VISIBLE_DEVICES=0 python3 IRES_DM.py --only_ires --prefix IRESDM_DirectedTraining --epochs 150 --cls_model ''
```

#### Parameters
```
Namespace(device='0', prefix='IRESDM_DirectedTraining', epochs=150, image_size=200, only_ires=True, batch_size=64, timesteps=50, beta_scheduler='linear', learning_rate=0.0001, global_seed=42, ema_beta=0.995, n_steps=10, n_samples=1000, save_and_sample_every=1, epochs_loss_show=5, time_warping=True, truncate_remain_right=True, utrlm_modelfile='../models/IRES_UTRLM_best_model', rnafm_modelfile='../models/IRES_RNAFM_best_model', cls_model='')
```

#### Generated Sequences
- `./results/IRESDM_DirectedTraining_Gene_Class1_correctNum6076_ep149.csv`

#### Saved Model
- `./models/IRESDM_DirectedTraining_best_model.pt`
please find in [link](https://drive.google.com/drive/folders/13U1VLNbTH75nqp_ieRUr6uOAp3IrJ8RU?usp=drive_link)

### 3.3 Post-processing and Filtering

#### Filtering Generated Sequences
Run the following notebooks to filter generated sequences:
- `./Script/IRESDM_Directed_Training_Filtering_Generated_Sequences.ipynb`
- `./Script/IRESDM_Reward_Guided_Filtering_Generated_Sequences.ipynb`

Filtering criteria:
- G/C content between 40% and 60%
- Absence of consecutive repeat nucleotides
- Lengths ranging from 160 nt to 200 nt
- IRES predicted probability exceeding 90%

#### Filtered Sequence Files
- `./results/FilterV1_IRESDM_RewardGuided_CLSutrlm_Gene_Class1_correctNum6178_ep149.csv`
- `./results/FilterV1_IRESDM_DirectedTraining_Gene_Class1_correctNum6076_ep149.csv`

#### Combining Filtered Sequences
Run the following notebook to select 2,000 sequences with minimal structural similarity to natural IRESs:
- `./Script/Combined_Filtered_Generated_Sequences_From_Two_Strategies.ipynb`

Combined sequence file:
- `./results/IRESDM_TwoStrategyCombined_Class1_GeneNum2000.csv`

#### Finding Similar Sequences
Run the following notebook to find the most similar IRES-DM generated sequence with Experimentally Validated IRES from IRESsite database:
- `./Script/FindSimilarSS_AmongIRESFrom_IRESsite_IRESDM.ipynb`

Results file:
- `IRESsite_shorter300bp_similarGenSeq.csv`

The index in columns:
- `RewardGuided_NearestSS_IRESsite_IRES`
- `RewardGuided_NearestSeq_IRESsite_IRES`
- `RewardGuided_MinSS_IRESsite_IRES`
- `RewardGuided_MinSeq_IRESsite_IRES`
- `DirectedTraining_NearestSS_IRESsite_IRES`
- `DirectedTraining_NearestSeq_IRESsite_IRES`
- `DirectedTraining_MinSS_IRESsite_IRES`
- `DirectedTraining_MinSeq_IRESsite_IRES`

can be linked to the first columns of:
- `IRESDM_DirectedTraining_Gene_Class1_correctNum6076_Length25+_ep149_SSMFE.csv`
- `IRESDM_RewardGuided_CLSutrlm_Gene_Class1_correctNum6178_Length25+_ep149_SSMFE.csv`

to find detailed information including sequences.

The Secondary Structure are stored in *.zip from [link](https://drive.google.com/drive/folders/1wGtaREfE-GxUDwn_attYob1DShZASQCO?usp=drive_link)

### 3.4 Running Time
For implementation, we trained both strategies for 150 epochs on a single NVIDIA Tesla GPU with 32GB memory. The reward-guided approach, which leveraged the IRES-UTRLM classifier for reward calculation, required approximately 352 seconds per epoch, while the direct training approach needed only 225 seconds per epoch. Both strategies initially generated 9,172 sequences to match the scale of our training dataset.
