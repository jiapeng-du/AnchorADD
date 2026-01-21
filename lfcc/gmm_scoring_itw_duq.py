from gmm_gduq import scoring, augment_ssi, augment_da7_series

scores_file = 'scores-lfcc-itw-a-duq-da5-da7.txt'
features = 'lfcc_duq'
dict_file = 'gmm_lfcc_asvspoof_itw_duq.pkl'

db_folder = '/root/autodl-tmp/datasets/release_in_the_wild/'  # e.g., '/path/to/ASVspoof_root/'

eval_folder = db_folder + 'wav/'
eval_ndx = db_folder + 'protocols/in_the_wild/in_the_wild.eval.trl.txt'
audio_ext = ''

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