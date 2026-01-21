# asvspoof2021_baseline.py —— 训练阶段使用 da5 + ΔUQ(LFCC_duq)
from gmm_gduq import train_gmm, augment_da5_series
from os.path import exists
import pickle

features = 'lfcc_duq'   # ← 启用 ΔUQ 特征
ncomp = 512
dict_file = 'gmm_19LA_lfcc_3.pkl'
dict_file_final = 'gmm_lfcc_asvspoof19_la_duq_x_final.pkl'

db_folder = '/root/autodl-tmp/datasets/ASVspoof2019_LA/'
train_folders = [db_folder + 'ASVspoof2019_LA_train/flac/', db_folder + 'ASVspoof2019_LA_dev/flac/']
train_keys = [db_folder + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt', db_folder + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt']
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

# 若你仍要兼容此前“init_partial → final”的组合方式，可保留下列合并逻辑（可选）
# gmm_dict = {}
# with open(dict_file + '_bonafide_init_partial.pkl', "rb") as tf:
#     gmm_dict['bona'] = pickle.load(tf)
# with open(dict_file + '_spoof_init_partial.pkl', "rb") as tf:
#     gmm_dict['spoof'] = pickle.load(tf)
# with open(dict_file_final, "wb") as f:
#     pickle.dump(gmm_dict, f)
