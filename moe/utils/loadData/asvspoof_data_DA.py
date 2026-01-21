import numpy as np
import soundfile as sf
import torch,os
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from .RawBoost import process_Rawboost_feature
import lightning as L
import subprocess

class asvspoof_dataModule(L.LightningDataModule):
        def __init__(self,args):
                super().__init__()
                self.args = args

                # TODO: change the dir to your own data dir
                # label file
                self.protocols_path = "./data/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/"
                self.train_protocols_file = self.protocols_path + "ASVspoof2019.LA.cm.train.trl.txt"
                self.dev_protocols_file = self.protocols_path + "ASVspoof2019.LA.cm.dev.trl.txt"
                # flac file dir
                self.dataset_base_path="./data/ASVspoof2019_LA/"
                self.train_set=self.dataset_base_path+"ASVspoof2019_LA_train/"
                self.dev_set=self.dataset_base_path+"ASVspoof2019_LA_dev/"
                # test set 
                self.eval_protocols_file_19 = self.protocols_path + "ASVspoof2019.LA.cm.eval.trl.txt"
                self.eval_set_19 = self.dataset_base_path+"ASVspoof2019_LA_eval/"
                self.eval_protocols_file_21 = "./data/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt"
                self.eval_set_21 = "./data/ASVspoof2021_LA_eval/"

                
                self.LA21 = "./data/aasist/datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt"
                self.LA21FLAC = "./data/aasist/datasets/ASVspoof2021_LA_eval/"
                self.LA21TRIAL = "./data/aasist/datasets/ASVspoof2021_LA_eval/CM_trial_metadata.txt"

                self.DF21 = "./data/aasist/datasets/ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt"
                self.DF21FLAC = "./data/aasist/datasets/ASVspoof2021_DF_eval/"
                self.DF21TRIAL = "./data/aasist/datasets/ASVspoof2021_DF_eval/trial_metadata.txt"

                self.ITWTXT = "./data/aasist/datasets/release_in_the_wild/label.txt"
                self.ITWDIR = "./data/aasist/datasets/release_in_the_wild/wav/"

                
                
                self.truncate = args.truncate
                self.predict = args.testset # LA21, DF21, ITW

        def setup(self, stage: str):
            # Assign train/val datasets for use in dataloaders
            if stage == "fit":
                d_label_trn, file_train = genSpoof_list(
                    dir_meta=self.train_protocols_file,
                    is_train=True,
                    is_eval=False
                )

                self.asvspoof19_trn_set = Dataset_ASVspoof2019_train(
                    list_IDs=file_train,
                    labels=d_label_trn,
                    base_dir=self.train_set,
                    cut=self.truncate,
                    args=self.args
                )

                _, file_dev = genSpoof_list(
                    dir_meta=self.dev_protocols_file,
                    is_train=False,
                    is_eval=False)

                self.asvspoof19_val_set = Dataset_ASVspoof2019_devNeval(
                    list_IDs=file_dev,
                    base_dir=self.dev_set,
                    args=self.args,
                    cut=self.truncate
                )

            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                file_eval = genSpoof_list(
                    dir_meta=self.eval_protocols_file_19,
                    is_train=False,
                    is_eval=True
                )
                self.asvspoof19_test_set = Dataset_ASVspoof2019_evaltest(
                    list_IDs=file_eval,
                    base_dir=self.eval_set_19,
                    cut=self.truncate
                )

            if stage == "predict":
                if self.predict == "LA21":
                    file_list = []
                    with open(self.LA21, 'r') as f:
                        l_meta = f.readlines()
                    for line in l_meta:
                        key = line.strip()
                        file_list.append(key)
                    print(f"no.{(len(file_list))} of eval  trials")
                    self.predict_set = Dataset_ASVspoof2019_evaltest(
                        list_IDs=file_list,
                        base_dir=self.LA21FLAC,
                        cut=self.truncate)

                elif self.predict == "DF21":
                    file_list = []
                    with open(self.DF21, 'r') as f:
                        l_meta = f.readlines()
                    for line in l_meta:
                        key = line.strip()
                        file_list.append(key)
                    print(f"no.{(len(file_list))} of eval  trials")
                    self.predict_set = Dataset_ASVspoof2019_evaltest(
                        list_IDs=file_list,
                        base_dir=self.DF21FLAC,
                        cut=self.truncate)

                elif self.predict == "ITW":
                    file_list = []
                    # 打开文件
                    with open(self.ITWTXT, 'r') as file:
                        lines = file.readlines()
                        for line in lines:
                            columns = line.split()
                            file_list.append(columns[1])
                    self.predict_set = dataset_itw(
                        list_IDs=file_list,
                        base_dir=self.ITWDIR,
                        cut=self.truncate)

        def train_dataloader(self):
            return DataLoader(self.asvspoof19_trn_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                              num_workers=4)

        def val_dataloader(self):
            return DataLoader(self.asvspoof19_val_set, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                              num_workers=4)

        def test_dataloader(self):
            datald = DataLoader(
                self.asvspoof19_test_set, batch_size=self.args.batch_size,
                shuffle=False, num_workers=4
            )
            if "," in self.args.gpuid:
                datald = DataLoader(
                    self.asvspoof19_test_set, batch_size=self.args.batch_size,
                    shuffle=False, num_workers=4,
                    sampler=DistributedSampler(self.asvspoof19_test_set)
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
            if "," in self.args.gpuid:
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


class dataset_itw(Dataset):
    def __init__(self, list_IDs, base_dir, cut=64600):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    # def __getitem__(self, index):
    #     key = self.list_IDs[index]
    #     X, _ = sf.read(os.path.join(self.base_dir, f"{key}.wav"))
    #     if self.cut == 0:
    #         X_pad = X
    #     else:
    #         X_pad = pad(X, self.cut)
    #     x_inp = Tensor(X_pad)
    #     return x_inp, key

    def __getitem__(self, index):
        key = self.list_IDs[index]
        file_path = os.path.join(self.base_dir, f"{key}")

        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"File missing: {file_path}")

        # 多重读取尝试
        readers = [
            self._read_with_soundfile,
            self._read_with_torchaudio,
            self._read_with_ffmpeg
        ]

        for reader in readers:
            try:
                X, _ = reader(file_path)
                if self.cut == 0:
                    X_pad = X
                else:
                    X_pad = pad(X, self.cut)
                x_inp = Tensor(X_pad)
                return x_inp, key
            except Exception as e:
                continue

    def _read_with_soundfile(self, path):
        return sf.read(path)

    def _read_with_torchaudio(self, path):
        waveform, _ = torchaudio.load(path)
        return waveform.mean(dim=0).numpy(), None

    def _read_with_ffmpeg(self, path):
        """通过ffmpeg读取音频"""
        import subprocess
        cmd = f"ffmpeg -i {path} -f wav - | python -c \"import sys; import numpy as np; from scipy.io import wavfile; print(wavfile.read(sys.stdin.buffer))\""
        proc = subprocess.run(cmd, shell=True, capture_output=True)
        fs = int(proc.stdout.split()[0])
        audio = np.frombuffer(proc.stdout.split()[1:], dtype=np.int16)
        return audio, fs

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            # key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, args, cut=64600):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.args = args
        self.cut = cut  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs = sf.read(os.path.join(self.base_dir, f"flac/{key}.flac"))
        if self.args.usingDA and (np.random.rand() < self.args.da_prob):
            X = process_Rawboost_feature(X, fs, self.args, self.args.algo)
        if self.cut == 0:
            X_pad = X
        else:
            X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        # 1. tensor 2.label 3.filename
        return x_inp, y, key


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir, args=None, cut=64600):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut  # take ~4 sec audio (64600 samples)
        self.args = args

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs = sf.read(os.path.join(self.base_dir, f"flac/{key}.flac"))
        if self.args.usingDA and ("ASVspoof2019_LA_dev" in self.base_dir):
            X = process_Rawboost_feature(X, fs, self.args, self.args.algo)
        if self.cut == 0:
            X_pad = X
        else:
            X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        # 1.tensor 2.filename
        return x_inp, key


class Dataset_ASVspoof2019_evaltest(Dataset):
    def __init__(self, list_IDs, base_dir, args=None, cut=64600):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut  # take ~4 sec audio (64600 samples)
        self.args = args

    def __len__(self):
        return len(self.list_IDs)

    # def __getitem__(self, index):
    #     key = self.list_IDs[index]
    #     X, fs = sf.read(os.path.join(self.base_dir,f"flac/{key}.flac"))
    #     if self.cut == 0:
    #         X_pad = X
    #     else:
    #         X_pad = pad(X, self.cut)
    #     x_inp = Tensor(X_pad)
    #     return x_inp, key

    def __getitem__(self, index):
        key = self.list_IDs[index]
        file_path = os.path.join(self.base_dir, f"flac/{key}.flac")

        # FFmpeg读取（保持原始特性）
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

            # 长度处理 - 完全等效于原始pad函数
            if self.cut > 0:
                x_len = len(X)
                if x_len >= self.cut:
                    X = X[:self.cut]
                else:
                    # 等效于原始pad函数的重复填充逻辑
                    num_repeats = self.cut // x_len + 1
                    X = np.tile(X, num_repeats)[:self.cut]
            else:
                # 如果cut=0，保持原始长度
                pass

            return torch.FloatTensor(X), key

        except Exception as e:
            raise RuntimeError(f"读取{file_path}失败: {str(e)}")
    # def __getitem__(self, index):
    #     key = self.list_IDs[index]
    #
    #     # 定义所有可能的文件路径（包含多种扩展名和子目录）
    #     possible_paths = [
    #         os.path.join(self.base_dir, f"flac/{key}.flac"),
    #         os.path.join(self.base_dir, f"flac/{key}.wav"),
    #         os.path.join(self.base_dir, f"wav/{key}.flac"),
    #         os.path.join(self.base_dir, f"wav/{key}.wav"),
    #         os.path.join(self.base_dir, f"{key}.flac"),
    #         os.path.join(self.base_dir, f"{key}.wav")
    #     ]
    #
    #     # 查找实际存在的文件路径
    #     file_path = None
    #     for path in possible_paths:
    #         if os.path.exists(path):
    #             file_path = path
    #             break
    #
    #     # 文件不存在直接报错
    #     if not file_path:
    #         raise FileNotFoundError(
    #             f"Audio file not found for key {key}. Tried paths:\n" +
    #             "\n".join(possible_paths)
    #         )
    #
    #     # 定义读取器优先级 - 针对FLAC问题优化
    #     readers = [
    #         self._read_with_ffmpeg,  # FFmpeg处理FLAC最可靠
    #         self._read_with_torchaudio,
    #         self._read_with_soundfile
    #     ]
    #
    #     last_exception = None
    #     for reader in readers:
    #         try:
    #             X, fs = reader(file_path)
    #
    #             # 确保音频数据是1D数组
    #             if X.ndim > 1:
    #                 X = X.mean(axis=0)
    #
    #             # 处理长度
    #             if self.cut > 0:
    #                 X = self._pad_or_trim(X)
    #
    #             return torch.Tensor(X), key
    #         except Exception as e:
    #             last_exception = e
    #             # 打印具体错误信息帮助调试
    #             print(f"Reader {reader.__name__} failed for {file_path}: {str(e)}")
    #             continue
    #
    #     # 所有读取方式都失败
    #     raise RuntimeError(
    #         f"All audio readers failed for file {file_path}\n"
    #         f"Last error: {str(last_exception)}"
    #     )
    #
    #
    # def _pad_or_trim(self, audio):
    #     """智能填充或裁剪音频"""
    #     if len(audio) > self.cut:
    #         # 随机裁剪
    #         start = np.random.randint(0, len(audio) - self.cut)
    #         return audio[start:start + self.cut]
    #     else:
    #         # 末端填充
    #         return np.pad(audio, (0, max(0, self.cut - len(audio))), mode='constant')
    #
    #
    # def _read_with_soundfile(self, path):
    #     """使用soundfile读取（优先尝试FLAC）"""
    #     try:
    #         # 显式指定格式为FLAC
    #         if path.endswith('.flac'):
    #             data, sr = sf.read(path, format='FLAC', dtype='float32')
    #         else:
    #             data, sr = sf.read(path, dtype='float32')
    #
    #         # 确保单声道
    #         if data.ndim > 1:
    #             data = data[:, 0] if data.shape[1] >= 1 else data.mean(axis=1)
    #         return data, sr
    #     except Exception as e:
    #         raise RuntimeError(f"Soundfile read failed: {str(e)}")
    #
    #
    # def _read_with_torchaudio(self, path):
    #     """使用torchaudio读取（优先尝试FLAC）"""
    #     try:
    #         # 显式指定FLAC格式
    #         if path.endswith('.flac'):
    #             waveform, sr = torchaudio.load(path, format='flac')
    #         else:
    #             waveform, sr = torchaudio.load(path)
    #
    #         # 转换为单声道并转为numpy
    #         if waveform.ndim > 1:
    #             waveform = waveform.mean(dim=0)
    #         return waveform.numpy(), sr
    #     except Exception as e:
    #         raise RuntimeError(f"Torchaudio read failed: {str(e)}")
    #
    #
    # def _read_with_ffmpeg(self, path):
    #     """通过ffmpeg读取音频（最健壮的实现）"""
    #     import subprocess
    #     import io
    #     from scipy.io import wavfile
    #
    #     try:
    #         # 使用内存缓冲避免临时文件
    #         cmd = [
    #             "ffmpeg",
    #             "-v", "error",  # 只显示错误
    #             "-i", path,
    #             "-f", "wav",
    #             "-acodec", "pcm_s16le",  # 确保标准格式
    #             "-ac", "1",  # 单声道
    #             "-ar", "16000",  # 标准采样率
    #             "-"  # 输出到stdout
    #         ]
    #
    #         # 运行FFmpeg并捕获输出
    #         proc = subprocess.run(
    #             cmd,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE,
    #             timeout=10  # 防止挂起
    #         )
    #
    #         # 检查返回码
    #         if proc.returncode != 0:
    #             error_msg = proc.stderr.decode('utf-8', 'ignore')
    #             raise RuntimeError(f"FFmpeg error ({proc.returncode}): {error_msg}")
    #
    #         # 检查是否有输出
    #         if not proc.stdout:
    #             raise RuntimeError("FFmpeg produced no output")
    #
    #         # 使用BytesIO创建内存文件
    #         audio_buffer = io.BytesIO(proc.stdout)
    #         sr, audio = wavfile.read(audio_buffer)
    #
    #         # 转换为浮点数 [-1, 1]
    #         if audio.dtype == np.int16:
    #             audio = audio.astype(np.float32) / 32768.0
    #         elif audio.dtype == np.int32:
    #             audio = audio.astype(np.float32) / 2147483648.0
    #         elif audio.dtype == np.uint8:  # 处理8位PCM
    #             audio = (audio.astype(np.float32) - 128) / 128.0
    #
    #         return audio, sr
    #     except Exception as e:
    #         raise RuntimeError(f"FFmpeg read failed: {str(e)}")
      
      
      
      
      
      
