# asvspoof2021_baseline.py —— 训练阶段使用 da5
from gmm import train_gmm, augment_da5_series
from os.path import exists
import pickle

features = 'lfcc'
ncomp = 512
dict_file = 'gmm_21DF_lfcc.pkl'
dict_file_final = 'gmm_lfcc_asvspoof21_df.pkl'

db_folder = '/root/autodl-tmp/datasets/ASVspoof2021_DF_eval/'
train_folder = db_folder + 'flac/'
train_folders = [train_folder for _ in range(2)]
train_keys = [
    db_folder + 'custom_protocols_21DF/ASVspoof2021_DF/asvspoof2021_df.train.trn.txt',
    db_folder + 'custom_protocols_21DF/ASVspoof2021_DF/asvspoof2021_df.dev.trl.txt'
]
audio_ext = '.flac'

# 训练 bona/spoof GMM（开启 da5 增强）
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

    gmm_dict = {'bona': gmm_bona._get_parameters(), 'spoof': gmm_spoof._get_parameters()}
    with open(dict_file, "wb") as tf:
        pickle.dump(gmm_dict, tf)

# —— 确保 final 文件与 dict_file 内容一致（不再依赖 gmm_bona/spoof 变量）——
with open(dict_file, "rb") as src:
    gmm_dict = pickle.load(src)
with open(dict_file_final, "wb") as dst:

    pickle.dump(gmm_dict, dst)
