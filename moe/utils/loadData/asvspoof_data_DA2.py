import numpy as np
import soundfile as sf
import torch, os
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from .RawBoost import process_Rawboost_feature
import lightning as L
import subprocess
        
class asvspoof_dataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset_name = args.dataset_name  # 新增：数据集名称参数

        # 根据数据集名称设置路径
        if self.dataset_name == "19":
            # 原有的19LA配置
            self.protocols_path = "/root/autodl-tmp/datasets/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/"
            self.train_protocols_file = self.protocols_path + "ASVspoof2019.LA.cm.train.trl.txt"
            self.dev_protocols_file = self.protocols_path + "ASVspoof2019.LA.cm.dev.trl.txt"
            self.eval_protocols_file = self.protocols_path + "ASVspoof2019.LA.cm.eval.trl.txt"
            self.train_set = "/root/autodl-tmp/datasets//ASVspoof2019_LA/ASVspoof2019_LA_train/"
            self.dev_set = "/root/autodl-tmp/datasets/ASVspoof2019_LA/ASVspoof2019_LA_dev/"
            self.eval_set = "/root/autodl-tmp/datasets/ASVspoof2019_LA/ASVspoof2019_LA_eval/"
            
        elif self.dataset_name == "LA21":
            # 21LA配置
            self.protocols_path = "/root/autodl-tmp/datasets/ASVspoof2021_LA_eval/custom_protocols_21LA/ASVspoof2021_LA/"
            self.train_protocols_file = self.protocols_path + "asvspoof2021_la.train.trn.txt"
            self.dev_protocols_file = self.protocols_path + "asvspoof2021_la.dev.trl.txt"
            self.eval_protocols_file = self.protocols_path + "asvspoof2021_la.eval.trl.txt"
            self.train_set = "/root/autodl-tmp/datasets/ASVspoof2021_LA_eval/"
            self.dev_set = "/root/autodl-tmp/datasets/ASVspoof2021_LA_eval/"
            self.eval_set = "/root/autodl-tmp/datasets/ASVspoof2021_LA_eval/"
            
        elif self.dataset_name == "DF21":
            # 21DF配置
            self.protocols_path = "/root/autodl-tmp/datasets/ASVspoof2021_DF_eval/custom_protocols_21DF/ASVspoof2021_DF/"
            self.train_protocols_file = self.protocols_path + "asvspoof2021_df.train.trn.txt"
            self.dev_protocols_file = self.protocols_path + "asvspoof2021_df.dev.trl.txt"
            self.eval_protocols_file = self.protocols_path + "asvspoof2021_df.eval.trl.txt"
            self.train_set = "/root/autodl-tmp/datasets/ASVspoof2021_DF_eval/"
            self.dev_set = "/root/autodl-tmp/datasets/ASVspoof2021_DF_eval/"
            self.eval_set = "/root/autodl-tmp/datasets/ASVspoof2021_DF_eval/"
            
        elif self.dataset_name == "ITW":
            # in_the_wild配置
            self.protocols_path = "/root/autodl-tmp/datasets/release_in_the_wild/protocols/in_the_wild/"
            self.train_protocols_file = self.protocols_path + "in_the_wild.train.trn.txt"
            self.dev_protocols_file = self.protocols_path + "in_the_wild.dev.trl.txt"
            self.eval_protocols_file = self.protocols_path + "in_the_wild.eval.trl.txt"
            self.train_set = "/root/autodl-tmp/datasets/release_in_the_wild/wav/"
            self.dev_set = "/root/autodl-tmp/datasets/release_in_the_wild/wav/"
            self.eval_set = "/root/autodl-tmp/datasets/release_in_the_wild/wav/"

        self.truncate = args.truncate
        self.predict = args.testset if hasattr(args, 'testset') else None

    def setup(self, stage: str):
        # 根据数据集名称和阶段设置数据
        if stage == "fit":
            # 训练集
            d_label_trn, file_train = genSpoof_list(
                dir_meta=self.train_protocols_file,
                is_train=True,
                is_eval=False
            )
            
            if self.dataset_name == "ITW":
                # ITW使用特殊的数据集类
                self.train_set = dataset_itw(
                    list_IDs=file_train,
                    labels=d_label_trn,
                    base_dir=self.train_set,
                    args=self.args,
                    cut=self.truncate
                )
            else:
                self.train_set = Dataset_ASVspoof2019_train(
                    list_IDs=file_train,
                    labels=d_label_trn,
                    base_dir=self.train_set,
                    cut=self.truncate,
                    args=self.args
                )

            # 验证集
            d_label_dev, file_dev = genSpoof_list(
                dir_meta=self.dev_protocols_file,
                is_train=False,
                is_eval=False
            )
            
            if self.dataset_name == "ITW":
                self.val_set = dataset_itw(
                    list_IDs=file_dev,
                    labels=d_label_dev,
                    base_dir=self.dev_set,
                    args=self.args,
                    cut=self.truncate
                )
            else:
                self.val_set = Dataset_ASVspoof2019_devNeval(
                    list_IDs=file_dev,
                    labels=d_label_dev,
                    base_dir=self.dev_set,
                    args=self.args,
                    cut=self.truncate
                )

        if stage == "test":
            # 测试集
            file_eval = genSpoof_list(
                dir_meta=self.eval_protocols_file,
                is_train=False,
                is_eval=True
            )
            
            if self.dataset_name == "ITW":
                self.test_set = dataset_itw(
                    list_IDs=file_eval,
                    labels={},
                    base_dir=self.eval_set,
                    args=self.args,
                    cut=self.truncate
                )
            else:
                self.test_set = Dataset_ASVspoof2019_evaltest(
                    list_IDs=file_eval,
                    labels={},
                    base_dir=self.eval_set,
                    args=self.args,
                    cut=self.truncate
                )

#        if stage == "predict":
#            # 预测集 - 保持原有逻辑
#            if self.predict == "LA21":
#                file_list = []
#                with open("./data/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt", 'r') as f:
#                    l_meta = f.readlines()
#                for line in l_meta:
#                    key = line.strip()
#                    file_list.append(key)
#                print(f"no.{(len(file_list))} of eval trials")
#                self.predict_set = Dataset_ASVspoof2019_evaltest(
#                    list_IDs=file_list,
#                    labels={},  # 预测集没有标签
#                    base_dir="./data/ASVspoof2021_LA_eval/",
#                    args=self.args,
#                    cut=self.truncate)
#
#            elif self.predict == "DF21":
#                file_list = []
#                with open("./data/aasist/datasets/ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt", 'r') as f:
#                    l_meta = f.readlines()
#                for line in l_meta:
#                    key = line.strip()
#                    file_list.append(key)
#                print(f"no.{(len(file_list))} of eval trials")
#                self.predict_set = Dataset_ASVspoof2019_evaltest(
#                    list_IDs=file_list,
#                    labels={},
#                    base_dir="./data/aasist/datasets/ASVspoof2021_DF_eval/",
#                    args=self.args,
#                    cut=self.truncate)
#
#            elif self.predict == "ITW":
#                file_list = []
#                with open("./data/aasist/datasets/release_in_the_wild/label.txt", 'r') as file:
#                    lines = file.readlines()
#                    for line in lines:
#                        columns = line.split()
#                        file_list.append(columns[1])
#                self.predict_set = dataset_itw(
#                    list_IDs=file_list,
#                    labels={},
#                    base_dir="./data/aasist/datasets/release_in_the_wild/wav/",
#                    args=self.args,
#                    cut=self.truncate)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    def test_dataloader(self):
        datald = DataLoader(
            self.test_set, batch_size=self.args.batch_size,
            shuffle=False, num_workers=4
        )
        if hasattr(self.args, 'gpuid') and "," in self.args.gpuid:
            datald = DataLoader(
                self.test_set, batch_size=self.args.batch_size,
                shuffle=False, num_workers=4,
                sampler=DistributedSampler(self.test_set)
            )
        return datald

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=4)
        if hasattr(self.args, 'gpuid') and "," in self.args.gpuid:
            predict_loader = DataLoader(
                self.predict_set,
                batch_size=self.args.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                sampler=DistributedSampler(self.predict_set),
                num_workers=4
            )
        return predict_loader

# 数据集类 - 统一使用FFmpeg读取
class dataset_itw(Dataset):
    def __init__(self, list_IDs, labels, base_dir, args, cut=64600):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = cut
        self.args = args

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        file_path = os.path.join(self.base_dir, f"{key}")
        
        x_tensor = read_audio_with_ffmpeg(file_path, self.cut, self.args)
        y = self.labels.get(key, 0)

        return x_tensor, y, key

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, args, cut=64600):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.args = args
        self.cut = cut

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        file_path = os.path.join(self.base_dir, f"flac/{key}.flac")
 
        
        # 使用统一的FFmpeg读取
        x_tensor = read_audio_with_ffmpeg(file_path, self.cut, self.args)
        y = self.labels[key]

        return x_tensor, y, key

class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, labels, base_dir, args=None, cut=64600):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = cut
        self.args = args

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        
        file_path = os.path.join(self.base_dir, f"flac/{key}.flac")
        
        # 使用统一的FFmpeg读取
        x_tensor = read_audio_with_ffmpeg(file_path, self.cut, self.args)
        y = self.labels[key]

        return x_tensor, y, key

class Dataset_ASVspoof2019_evaltest(Dataset):
    def __init__(self, list_IDs, labels, base_dir, args, cut=64600):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = cut
        self.args = args

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        
        file_path = os.path.join(self.base_dir, f"flac/{key}.flac")
        
        # 使用统一的FFmpeg读取
        x_tensor = read_audio_with_ffmpeg(file_path, self.cut, self.args)
        y = self.labels.get(key, 0)

        return x_tensor, y, key


# 辅助函数保持不变
def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, key, _, _, label = parts
                file_list.append(key)
                d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, key, _, _, _ = parts
                file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, key, _, _, label = parts
                file_list.append(key)
                d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


def read_audio_with_ffmpeg(file_path, cut=64600, args=None):
    """统一的FFmpeg音频读取函数"""
    try:
        # 获取原始音频信息
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe失败: {result.stderr}")

        sr, channels = map(int, result.stdout.strip().split())

        # 读取音频数据（保持原始格式）
        read_cmd = [
            "ffmpeg", "-v", "error",
            "-i", file_path,
            "-f", "f32le",  # 原始float32格式输出
            "-acodec", "pcm_f32le",
            "-ar", str(sr),  # 保持原始采样率
            "-ac", str(channels),  # 保持原始声道数
            "-"  # 输出到stdout
        ]
        result = subprocess.run(read_cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg失败: {result.stderr.decode('utf-8', 'ignore')}")

        # 处理多声道（如需单声道则取均值）
        X = np.frombuffer(result.stdout, dtype=np.float32)
        if channels > 1:
            X = X.reshape(-1, channels).mean(axis=1)

         		# 数据增强 - 仅在训练模式下应用
        if args.usingDA:
            if args.algo == 5:
	            if np.random.rand() < args.da_prob:
		            try:
		                X = process_Rawboost_feature(X, sr, args, args.algo)
		            except Exception as e:
		                print(f"RawBoost处理失败 {file_path}: {str(e)}，跳过数据增强")
		                # 继续使用原始音频数据
            else:
	#            X = process_Rawboost_feature(X, sr, args, args.algo)
	            try:
		            X = process_Rawboost_feature(X, sr, args, args.algo)
	            except Exception as e:
		            print(f"RawBoost处理失败 {file_path}: {str(e)}，跳过数据增强")
		            # 继续使用原始音频数据
        # 长度处理
        if cut > 0:
            x_len = len(X)
            if x_len >= cut:
                X = X[:cut]
            else:
                num_repeats = cut // x_len + 1
                X = np.tile(X, num_repeats)[:cut]

        x_tensor = torch.as_tensor(X.copy(), dtype=torch.float32)
        return x_tensor
        
    except Exception as e:
        raise RuntimeError(f"读取{file_path}失败: {str(e)}")