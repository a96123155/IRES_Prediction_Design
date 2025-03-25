# python3 write_to_fasta.py --filename vJun8_Fold0_v10.3_UTRLM1.4_FinetuneESM_lr1e-4_dr5_bos_CLS20_10folds_F0_AvgEmbFalse_BosEmbTrue_epoch100_nodes40_dropout30.5_finetuneESMTrue_finetuneLastLayerESMFalse_lr0.0001_test.csv --column_name sequence --output_filename vJun8_Fold0_v10.3_UTRLM1.4_FinetuneESM_lr1e-4_dr5_bos_CLS20_10folds_F0_AvgEmbFalse_BosEmbTrue_epoch100_nodes40_dropout30.5_finetuneESMTrue_finetuneLastLayerESMFalse_lr0.0001_test.fa
import argparse
import pandas as pd
from tqdm import tqdm

def write_to_fasta(filename, column_name, output_filename):
    # 读取 CSV 文件
    df = pd.read_csv(filename)
    
    # 提取指定列的序列
    seqs = list(df[column_name])
    
    # 初始化 FASTA 格式字符串
    fa = ""
    for s in tqdm(seqs):
        fa += f'>{s}\n{s}\n'
    
    # 将 FASTA 格式字符串写入文件
    with open(output_filename, 'w') as f:
        f.write(fa[:-1])

if __name__ == "__main__":
    # 设置参数解析器
    parser = argparse.ArgumentParser(description="Convert sequences from a CSV file to FASTA format.")
    parser.add_argument('--filename', type=str, help='The input CSV file.')
    parser.add_argument('--column_name', type=str, default='sequence', help='The column name containing sequences.')
    parser.add_argument('--output_filename', type=str, help='The output FASTA file.')
    
    args = parser.parse_args()
    
    # 调用函数进行转换
    write_to_fasta(args.filename, args.column_name, args.output_filename)