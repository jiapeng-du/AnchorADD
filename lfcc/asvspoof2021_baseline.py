from gmm import train_gmm, augment_da5_series
from os.path import exists
import pickle

features = 'lfcc'   
ncomp = 512
dict_file = 'gmm_LA_lfcc_2021.pkl'
dict_file_final = 'gmm_lfcc_asvspoof21_la.pkl'

db_folder = '/root/autodl-tmp/datasets/ASVspoof2021_LA_eval/'
train_folder = db_folder + 'flac/'
train_folders = [train_folder for _ in range(2)]
train_keys = [
    db_folder + 'custom_protocols_21LA/ASVspoof2021_LA/asvspoof2021_la.train.trn.txt',
    db_folder + 'custom_protocols_21LA/ASVspoof2021_LA/asvspoof2021_la.dev.trl.txt'
]
audio_ext = '.flac'

# 训练 bona/spoof GMM（开启 da5 增强；建议 init_only=False 直接跑完整 EM）
if not exists(dict_file):
    gmm_bona = train_gmm(
        data_label='bonafide',
        features=features,
        train_keys=train_keys,
        train_folders=train_folders,
        audio_ext=audio_ext,
        dict_file=dict_file,
        ncomp=ncomp,
        init_only=False,
        augment_fn=lambda sig, fs: augment_da5_series(sig, fs)  # da5
    )
    gmm_spoof = train_gmm(
        data_label='spoof',
        features=features,
        train_keys=train_keys,
        train_folders=train_folders,
        audio_ext=audio_ext,
        dict_file=dict_file,
        ncomp=ncomp,
        init_only=False,
        augment_fn=lambda sig, fs: augment_da5_series(sig, fs)  # da5
    )

    # 直接保存最终参数，评估脚本读取这个文件
    with open(dict_file_final, "wb") as tf:
        pickle.dump({'bona': gmm_bona._get_parameters(),
                     'spoof': gmm_spoof._get_parameters()}, tf)