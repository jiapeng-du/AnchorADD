# gmm.py —— LFCC(+ΔUQ)+GMM, with Rawboost-style DA hooks (da5 in train, da3 in eval)

from numpy import log, exp, infty, zeros_like, vstack, zeros, errstate, finfo, sqrt, floor, tile, concatenate
from numpy import arange, meshgrid, ceil, linspace
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
from scipy.signal import lfilter, iirnotch, filtfilt
from LFCC_pipeline import lfcc
from scipy.fft import rfft, irfft
from os.path import exists
from random import sample, randint, random
import soundfile as sf
import logging
import pandas
import pickle
import math
import h5py
import numpy as np
import hashlib

# ── logging ──
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# ───────────────────────────────────────────────────────────────────────────────
#                           Data Augmentation (Rawboost-like)
#   algo: 1=LnL, 2=ISD, 3=SSI, 5=series(1+2)
# ───────────────────────────────────────────────────────────────────────────────

def augment_lnl(sig, fs, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000):
    x = sig.astype(np.float32)
    for _ in range(nBands):
        f0 = np.random.uniform(minF, min(maxF, fs/2 - 50.0))
        bw = np.random.uniform(minBW, maxBW)
        Q = f0 / max(1.0, bw)
        try:
            b, a = iirnotch(w0=f0/(fs/2.0), Q=Q)
            x = filtfilt(b, a, x)
        except Exception:
            pass
    return x

def augment_isd(sig, fs, P=10, g_sd=2):
    x = sig.astype(np.float32).copy()
    N = len(x)
    num_impulses = int(N * P / 100.0 / 10) + 1
    for _ in range(num_impulses):
        pos = randint(0, max(0, N-1))
        span = randint(1, max(1, int(0.002 * fs)))  # ~2ms
        end = min(N, pos + span)
        noise = np.random.randn(end - pos).astype(np.float32) * g_sd * 1e-3
        x[pos:end] += noise
    return x

def _apply_colored_noise(x, fs, snr_db, color='pink'):
    N = len(x)
    n = np.random.randn(N).astype(np.float32)
    if color in ('pink', 'brown'):
        X = rfft(n)
        freqs = np.linspace(1, N//2+1, len(X))
        H = 1.0 / np.sqrt(freqs) if color == 'pink' else 1.0 / (freqs)
        X_col = X * H
        n = irfft(X_col, n=N).astype(np.float32)
    Px = np.mean(x**2) + 1e-12
    Pn = np.mean(n**2) + 1e-12
    alpha = math.sqrt(Px / (Pn * (10**(snr_db/10.0))))
    n = n * alpha
    return x + n

def augment_ssi(sig, fs, SNRmin=10, SNRmax=40, color='pink'):
    snr = np.random.uniform(SNRmin, SNRmax)
    return _apply_colored_noise(sig.astype(np.float32), fs, snr_db=snr, color=color)

def augment_da5_series(sig, fs,
                       nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
                       P=10, g_sd=2):
    x = augment_lnl(sig, fs, nBands, minF, maxF, minBW, maxBW)
    x = augment_isd(x, fs, P=P, g_sd=g_sd)
    return x
def augment_da7_series(sig, fs, P=10, g_sd=2, SNRmin=10, SNRmax=40, color='pink'):
    """da7: series(2+3) → 先 ISD 稀疏加噪，再 SSI 有色噪声（与 main.py 中 algo=7 含义一致）。"""
    x = augment_isd(sig, fs, P=P, g_sd=g_sd)
    x = augment_ssi(x, fs, SNRmin=SNRmin, SNRmax=SNRmax, color=color)
    return x

# ───────────────────────────────────────────────────────────────────────────────
#                               Feature Extraction
# ───────────────────────────────────────────────────────────────────────────────

def Deltas(x, width=3):
    hlen = int(floor(width/2))
    win = list(range(hlen, -hlen-1, -1))
    xx_1 = tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx)
    return D[:, hlen*2:]

def _det_seed_from_path(path_str: str) -> int:
    return int(hashlib.md5(path_str.encode('utf-8')).hexdigest(), 16) % (2**32)

def _lfcc_core(sig, fs, num_ceps=20, low_freq=0, high_freq=4000, order_deltas=2):
    lfccs = lfcc(sig=sig, fs=fs, num_ceps=num_ceps, low_freq=low_freq, high_freq=high_freq).T
    if order_deltas > 0:
        feats = [lfccs]
        for _ in range(order_deltas):
            feats.append(Deltas(feats[-1]))
        lfccs = vstack(feats)
    return lfccs  # shape: (F, T)

def extract_lfcc(file, num_ceps=20, order_deltas=2, low_freq=0, high_freq=4000, augment=None):
    sig, fs = sf.read(file)
    if augment is not None:
        try:
            sig = augment(sig, fs)
        except Exception as e:
            logging.warning(f"Augment failed on {file}: {e}")
    return _lfcc_core(sig, fs, num_ceps=num_ceps, low_freq=low_freq, high_freq=high_freq, order_deltas=order_deltas)

def extract_lfcc_duq(file, num_ceps=20, order_deltas=2, low_freq=0, high_freq=4000,
                     n_anchor_frames=20, augment=None):
    """
    ΔUQ: 在帧级 LFCC 上做“锚点均值差分”，输出 [X ; X - mean_anchor]（维度翻倍）。
    - anchor 来自同一条语音的若干帧；按文件名固定随机种子，保证稳定复现。
    """
    sig, fs = sf.read(file)
    if augment is not None:
        try:
            sig = augment(sig, fs)
        except Exception as e:
            logging.warning(f"Augment failed on {file}: {e}")
    X = _lfcc_core(sig, fs, num_ceps=num_ceps, low_freq=low_freq, high_freq=high_freq, order_deltas=order_deltas)  # (F,T)
    T = X.shape[1]
    if T == 0:
        return X
    rng = np.random.RandomState(_det_seed_from_path(file))
    idx = rng.choice(T, size=min(n_anchor_frames, T), replace=False)
    A_mean = np.mean(X[:, idx], axis=1, keepdims=True)  # (F,1)
    X_delta = X - A_mean
    
    # 将 A_mean 复制成与 X_delta 相同的列数
    A_mean_tiled = np.tile(A_mean, (1, T))  # (F, T)
    
    return vstack([A_mean_tiled, X_delta])  # (2F, T)

def extract_features(file, features, cached=False, augment=None):
    def get_feats():
        if features == 'lfcc':
            return extract_lfcc(file, augment=augment)
        elif features == 'lfcc_duq':
            return extract_lfcc_duq(file, augment=augment)
        else:
            return None

    # 有增强时不要写通用缓存（增强是随机/条件性）；无增强则缓存以加速
    if cached and augment is None:
        cache_file = features + '.h5'
        h5 = h5py.File(cache_file, 'a')
        group = h5.get(file)
        if group is None:
            data = get_feats()
            h5.create_dataset(file, data=data, compression='gzip')
        else:
            data = group[()]
        h5.close()
        return data
    else:
        return get_feats()
#def extract_features(file, features, cached=False, augment=None):
#    def get_feats():
#        if features == 'lfcc':
#            return extract_lfcc(file, augment=augment)
#        elif features == 'lfcc_duq':
 #           return extract_lfcc_duq(file, augment=augment)
#        else:
#            return None

    # 注意：
    # 1）有数据增强时不缓存（augment is not None）
    # 2）对 lfcc_duq 直接禁用缓存，避免 h5 太大撑爆磁盘
    if cached and augment is None and features != 'lfcc_duq':
        cache_file = os.path.join(FEATURE_CACHE_DIR, features + '.h5')
        h5 = h5py.File(cache_file, 'a')
        if file in h5:
            data = h5[file][()]
        else:
            data = get_feats()
            h5.create_dataset(file, data=data, compression='gzip')
        h5.close()
        return data
    else:
        # 不用缓存：直接算一次返回
        return get_feats()


# ───────────────────────────────────────────────────────────────────────────────
#                                   GMM train / score
# ───────────────────────────────────────────────────────────────────────────────

def train_gmm(data_label, features, train_keys, train_folders, audio_ext, dict_file, ncomp, init_only=False,
              augment_fn=None):
    logging.info('Start GMM training.')

    partial_gmm_dict_file = '_'.join((dict_file, data_label, 'init', 'partial.pkl'))
    if exists(partial_gmm_dict_file):
        gmm = GaussianMixture(covariance_type='diag')
        with open(partial_gmm_dict_file, "rb") as tf:
            gmm._set_parameters(pickle.load(tf))
    else:
        data = list()
        for k, train_key in enumerate(train_keys):
            pd = pandas.read_csv(train_key, sep=' ', header=None)
            files = pd[pd[4] == data_label][1]
            files_subset = (files.reset_index()[1]).loc[list(range(0, len(files), 10))]  # 十分抽样做初始化
            for file in files_subset:
                Tx = extract_features(train_folders[k] + file + audio_ext,
                                      features=features,
                                      cached=(augment_fn is None),
                                      augment=augment_fn)
                data.append(Tx.T)
        X = vstack(data)
        gmm = GaussianMixture(n_components=ncomp,
                              random_state=None,
                              covariance_type='diag',
                              max_iter=10,
                              verbose=2,
                              verbose_interval=1).fit(X)
        logging.info('GMM init done - llh: %.5f' % gmm.lower_bound_)
        with open(partial_gmm_dict_file, "wb") as f:
            pickle.dump(gmm._get_parameters(), f)

    if init_only:
        return gmm

    # 手写 EM（注意：这里默认沿用 augment_fn；若想稳定收敛，可在此改为 augment=None）
    prev_lower_bound = -infty
    for i in range(10):
        partial_gmm_dict_file = '_'.join((dict_file, data_label, str(i), 'partial.pkl'))
        if exists(partial_gmm_dict_file):
            with open(partial_gmm_dict_file, "rb") as tf:
                gmm._set_parameters(pickle.load(tf))
                continue

        nk_acc = zeros_like(gmm.weights_)
        mu_acc = zeros_like(gmm.means_)
        sigma_acc = zeros_like(gmm.covariances_)
        log_prob_norm_acc = 0
        n_samples = 0
        for k, train_key in enumerate(train_keys):
            pd = pandas.read_csv(train_key, sep=' ', header=None)
            files = pd[pd[4] == data_label][1]
            for file in files.values:
                Tx = extract_features(train_folders[k] + file + audio_ext,
                                      features=features,
                                      cached=(augment_fn is None),
                                      augment=augment_fn)
                n_samples += Tx.shape[1]

                weighted_log_prob = gmm._estimate_weighted_log_prob(Tx.T)
                log_prob_norm = logsumexp(weighted_log_prob, axis=1)
                with errstate(under='ignore'):
                    log_resp = weighted_log_prob - log_prob_norm[:, None]
                log_prob_norm_acc += log_prob_norm.sum()

                resp = exp(log_resp)
                nk_acc += resp.sum(axis=0) + 10 * finfo(log(1).dtype).eps
                mu_acc += resp.T @ Tx.T
                sigma_acc += resp.T @ (Tx.T ** 2)

        gmm.means_ = mu_acc / nk_acc[:, None]
        # 方差地板，提升数值稳健
        var_floor = 1e-5
        gmm.covariances_ = np.maximum(sigma_acc / nk_acc[:, None] - gmm.means_ ** 2 + gmm.reg_covar, var_floor)
        gmm.weights_ = nk_acc / n_samples
        gmm.weights_ /= gmm.weights_.sum()
        gmm.precisions_cholesky_ = 1. / sqrt(gmm.covariances_)

        with open(partial_gmm_dict_file, "wb") as f:
            pickle.dump(gmm._get_parameters(), f)

        lower_bound = log_prob_norm_acc / n_samples
        change = lower_bound - prev_lower_bound
        logging.info("  Iteration %d\t llh %.5f\t ll change %.5f" % (i, lower_bound, change))
        prev_lower_bound = lower_bound
        if abs(change) < gmm.tol:
            logging.info('  Converged; too small change')
            gmm.converged_ = True
            break

    return gmm

def scoring(scores_file, dict_file, features, eval_ndx, eval_folder, audio_ext,
            features_cached=True, flag_debug=False, augment_fn=None):
    logging.info('Scoring eval data')

    gmm_bona = GaussianMixture(covariance_type='diag')
    gmm_spoof = GaussianMixture(covariance_type='diag')
    with open(dict_file, "rb") as tf:
        gmm_dict = pickle.load(tf)
        gmm_bona._set_parameters(gmm_dict['bona'])
        gmm_spoof._set_parameters(gmm_dict['spoof'])

    pd = pandas.read_csv(eval_ndx, sep=' ', header=None)
    if flag_debug:
        pd = pd[:1000]

    files = pd[1].values
    scr = zeros_like(files, dtype=log(1).dtype)
    for i, file in enumerate(files):
        if (i+1) % 1000 == 0:
            logging.info("\t...%d/%d..." % (i+1, len(files)))
        try:
            Tx = extract_features(eval_folder + file + audio_ext,
                                  features=features,
                                  cached=(features_cached and augment_fn is None),
                                  augment=augment_fn)
            scr[i] = gmm_bona.score(Tx.T) - gmm_spoof.score(Tx.T)
        except Exception as e:
            logging.warning(e)
            scr[i] = log(1)

    pd_out = pandas.DataFrame({'files': files, 'scores': scr})
    pd_out.to_csv(scores_file, sep=' ', header=False, index=False)
    logging.info('\t... scoring completed.\n')
