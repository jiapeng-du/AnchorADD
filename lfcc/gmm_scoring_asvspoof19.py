# gmm_scoring_asvspoof21.py —— 评估阶段使用 da3
from gmm import scoring, augment_ssi

scores_file = 'scores-lfcc-asvspoof19-LA-da3.txt'
features = 'lfcc'
dict_file = 'gmm_lfcc_asvspoof19_la.pkl'

db_folder = '/root/autodl-tmp/datasets/ASVspoof2019_LA/'  # e.g., '/path/to/ASVspoof_root/'

eval_folder = db_folder + 'ASVspoof2019_LA_eval/flac/'
eval_ndx = db_folder + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
audio_ext = '.flac'

scoring(
    scores_file=scores_file,
    dict_file=dict_file,
    features=features,
    eval_ndx=eval_ndx,
    eval_folder=eval_folder,
    audio_ext=audio_ext,
    features_cached=True,        # 有增强时，内部会自动禁用缓存写入，避免污染
    augment_fn=lambda sig, fs: augment_ssi(sig, fs, SNRmin=10, SNRmax=40)  # da3
)








