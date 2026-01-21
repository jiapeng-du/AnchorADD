# itw_baseline_duq.py —— 训练阶段使用 da5 + ΔUQ (LFCC_duq)
from gmm_gduq import train_gmm, augment_da5_series
from os.path import exists
import pickle

# =========================
# 基本配置
# =========================
features = 'lfcc_duq'          # 启用 ΔUQ 特征
ncomp = 512

# 中间文件（可选，仅用于防止重复训练）
dict_file = 'gmm_ITW_lfcc_duq.pkl'

# 最终模型文件（评估脚本只读这个）
dict_file_final = 'gmm_lfcc_asvspoof_itw_duq.pkl'

# 数据路径
db_folder = '/root/autodl-tmp/datasets/release_in_the_wild/'
train_folder = db_folder + 'wav/'
train_folders = [train_folder for _ in range(2)]

train_keys = [
    db_folder + 'protocols/in_the_wild/in_the_wild.train.trn.txt',
    db_folder + 'protocols/in_the_wild/in_the_wild.dev.trl.txt'
]

audio_ext = ''   # ITW 是 wav，无后缀

# =========================
# 训练 GMM（da5 + ΔUQ）
# =========================
# 逻辑：
# - init_only=False：完整 EM
# - augment_fn=da5：与论文 / cul-eer21 设置一致
# - 最终参数直接来自 gmm._get_parameters()
# =========================

if not exists(dict_file_final):

    print(">>> Training bonafide GMM (da5 + ΔUQ)")
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

    print(">>> Training spoof GMM (da5 + ΔUQ)")
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

    # =========================
    # 保存最终模型（唯一权威来源）
    # =========================
    gmm_dict = {
        'bona': gmm_bona._get_parameters(),
        'spoof': gmm_spoof._get_parameters()
    }

    with open(dict_file_final, "wb") as f:
        pickle.dump(gmm_dict, f)

    print(f"✅ Final GMM model saved to: {dict_file_final}")

else:
    print(f"✔ Final model already exists: {dict_file_final}")
