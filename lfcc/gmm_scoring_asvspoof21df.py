# gmm_scoring_asvspoof21.py —— 评估阶段使用 da3/da7
from gmm import scoring, augment_ssi, augment_da7_series

scores_file = 'scores-lfcc-asvspoof21-DF-da7.txt'
features = 'lfcc'
dict_file = 'gmm_lfcc_asvspoof21_df.pkl'


db_folder = '/root/autodl-tmp/datasets/ASVspoof2021_DF_eval/'
eval_folder = db_folder + 'flac/'

# ⚠️ 这里可先改成 dev 再改回 eval
eval_ndx = db_folder + 'custom_protocols_21DF/ASVspoof2021_DF/asvspoof2021_df.eval.trl.txt'
audio_ext = '.flac'

scoring(
    scores_file=scores_file,
    dict_file=dict_file,
    features=features,
    eval_ndx=eval_ndx,
    eval_folder=eval_folder,
    audio_ext=audio_ext,
    features_cached=True,        # 有增强时，内部会自动禁用缓存写入，避免污染
    augment_fn=lambda sig, fs: augment_da7_series(sig, fs, P=10, g_sd=2, SNRmin=10, SNRmax=40)  # da7
)