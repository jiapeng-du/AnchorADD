# split_datasets_fixed.py
import os
import random
import argparse
from collections import defaultdict

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)

def read_2021_la_metadata(meta_file):
    """读取ASVspoof2021 LA元数据"""
    samples = []
    with open(meta_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
#                speaker_id = parts[0]
                filename = parts[1]
                label = parts[5]  # spoof or bonafide
                # 保留所有样本
                samples.append(( filename, label))
    return samples

def read_2021_df_metadata(meta_file):
    """读取ASVspoof2021 DF元数据"""
    samples = []
    with open(meta_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
#                speaker_id = parts[0]
                filename = parts[1]
                label = parts[5]  # spoof or bonafide
                # 保留所有样本
                samples.append(( filename, label))
    return samples

#def read_in_the_wild_metadata(meta_file):
#    """读取in_the_wild元数据"""
#    samples = []
#    with open(meta_file, 'r') as f:
#        for line in f:
#            parts = line.strip().split()
#            if len(parts) >= 5:
##                utt_id = parts[0]
#                audio_file = parts[1]
#                label = parts[4]  # spoof or bonafide
#                # 修复ITW标签问题
##                if label == "bona-fide":
##                    label = "bonafide"
##                elif label == "spoof":
##                    label = "spoof"
#                # 如果还有其他变体，可以在这里添加
#                samples.append(( audio_file, label))
#    return samples

def read_in_the_wild_metadata(meta_file):
    """读取ASVspoof2021 DF元数据"""
    samples = []
    with open(meta_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:  # 最少需要4部分：uttID, 文件名, -, 标签
                filename = parts[1]  # 第二列：序号/文件名
                label = parts[-1]    # 最后一列：spoof 或 bonafide
                samples.append((filename, label))
    return samples

def split_dataset(samples, dataset_name, output_dir, ratios=(0.2, 0.2, 0.6)):
    """划分数据集为train/dev/eval - 修复版本"""
    
    # 按标签分组
    spoof_samples = [s for s in samples if s[1] == 'spoof']
    bonafide_samples = [s for s in samples if s[1] == 'bonafide']
    
    print(f"{dataset_name} - Spoof samples: {len(spoof_samples)}, Bonafide samples: {len(bonafide_samples)}, Total samples: {len(samples)}")
    
    # 随机打乱
    random.shuffle(spoof_samples)
    random.shuffle(bonafide_samples)
    
    # 计算划分数量 - 修复整数截断问题
    train_ratio, dev_ratio, eval_ratio = ratios
    
    # 计算每个类别的划分数量
    def calculate_splits(total, ratios):
        train_count = int(total * ratios[0])
        dev_count = int(total * ratios[1])
        # 剩余的全部分配给eval，确保总数不变
        eval_count = total - train_count - dev_count
        return train_count, dev_count, eval_count
    
    spoof_train, spoof_dev, spoof_eval = calculate_splits(len(spoof_samples), ratios)
    bonafide_train, bonafide_dev, bonafide_eval = calculate_splits(len(bonafide_samples), ratios)
    
    print(f"Split - Spoof: train={spoof_train}, dev={spoof_dev}, eval={spoof_eval}")
    print(f"Split - Bonafide: train={bonafide_train}, dev={bonafide_dev}, eval={bonafide_eval}")
    
    # 划分样本
    train_samples = (spoof_samples[:spoof_train] + 
                    bonafide_samples[:bonafide_train])
    dev_samples = (spoof_samples[spoof_train:spoof_train+spoof_dev] + 
                  bonafide_samples[bonafide_train:bonafide_train+bonafide_dev])
    eval_samples = (spoof_samples[spoof_train+spoof_dev:] + 
                   bonafide_samples[bonafide_train+bonafide_dev:])
    
    # 验证总数是否正确
    total_after_split = len(train_samples) + len(dev_samples) + len(eval_samples)
    if total_after_split != len(samples):
        print(f"WARNING: Sample count mismatch! Before: {len(samples)}, After: {total_after_split}")
    
    # 再次打乱每个集合
    random.shuffle(train_samples)
    random.shuffle(dev_samples)
    random.shuffle(eval_samples)
    
    # 写入文件
    def write_protocol(samples, filename, set_name):
        with open(filename, 'w') as f:
            for  audio_file, label in samples:
                # 格式: - filename - - label
                f.write(f"- {audio_file} - - {label}\n")
        print(f"  {set_name}: {len(samples)} samples -> {filename}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    prefix = dataset_name.lower().replace(' ', '_')
    write_protocol(train_samples, os.path.join(output_dir, f'{prefix}.train.trn.txt'), 'Train')
    write_protocol(dev_samples, os.path.join(output_dir, f'{prefix}.dev.trl.txt'), 'Dev')
    write_protocol(eval_samples, os.path.join(output_dir, f'{prefix}.eval.trl.txt'), 'Eval')
    
    # 打印验证信息
    print(f"  Verification: {len(train_samples)} + {len(dev_samples)} + {len(eval_samples)} = {len(train_samples) + len(dev_samples) + len(eval_samples)}")
    
    return len(train_samples), len(dev_samples), len(eval_samples)

def main():
    parser = argparse.ArgumentParser(description='Split datasets into train/dev/eval')
    parser.add_argument('--la_meta', type=str, default = '/data/pytorch_lightning_FAD-main/data/aasist/datasets/ASVspoof2021_LA_eval/CM_trial_metadata.txt', 
                       help='Path to ASVspoof2021 LA metadata file (CM_trial_metadata.txt)')
    parser.add_argument('--df_meta', type=str, default = '/data/pytorch_lightning_FAD-main/data/aasist/datasets/ASVspoof2021_DF_eval/trial_metadata.txt',
                       help='Path to ASVspoof2021 DF metadata file (trial_metadata.txt)')
    parser.add_argument('--wild_meta', type=str, default = '/data/pytorch_lightning_FAD-main/data/aasist/datasets/release_in_the_wild/meta.txt',
                       help='Path to in_the_wild metadata file (meta.txt)')
    parser.add_argument('--output_dir', type=str, default='/data/pytorch_lightning_FAD-main/data/protocols',
                       help='Output directory for protocol files')
    parser.add_argument('--ratios', type=float, nargs=3, default=[0.2, 0.2, 0.6],
                       help='Train/Dev/Eval ratios (default: 0.2 0.2 0.6)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Dataset Splitting Tool - FIXED VERSION")
    print(f"Ratios: Train={args.ratios[0]}, Dev={args.ratios[1]}, Eval={args.ratios[2]}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    # 处理ASVspoof2021 LA
    print("\n1. Processing ASVspoof2021 LA...")
    la_samples = read_2021_la_metadata(args.la_meta)
    la_output_dir = os.path.join(args.output_dir, 'ASVspoof2021_LA')
    split_dataset(la_samples, "ASVspoof2021 LA", la_output_dir, args.ratios)
    
    # 处理ASVspoof2021 DF
    print("\n2. Processing ASVspoof2021 DF...")
    df_samples = read_2021_df_metadata(args.df_meta)
    df_output_dir = os.path.join(args.output_dir, 'ASVspoof2021_DF')
    split_dataset(df_samples, "ASVspoof2021 DF", df_output_dir, args.ratios)
    
    # 处理in_the_wild
    print("\n3. Processing In The Wild...")
    wild_samples = read_in_the_wild_metadata(args.wild_meta)
    wild_output_dir = os.path.join(args.output_dir, 'in_the_wild')
    split_dataset(wild_samples, "In The Wild", wild_output_dir, args.ratios)
    
    print("\n" + "=" * 60)
    print("All datasets have been successfully split!")
    print(f"Protocol files saved to: {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()