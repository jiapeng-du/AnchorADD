# gmm.py —— LFCC+GMM with Rawboost-style DA hooks (da5 in train, da3 in eval)
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

# ── logging ──
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# ───────────────────────────────────────────────────────────────────────────────
#                           Data Augmentation (Rawboost-like)
#   Ref to main.py algo mapping: 1=LnL_convolutive, 2=ISD_additive, 3=SSI_additive, 5=series(1+2)
#   We implement lightweight equivalents guided by main.py’s hyperparams.  :contentReference[oaicite:1]{index=1}
# ───────────────────────────────────────────────────────────────────────────────

def augment_lnl(sig, fs, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000):
    """da1: LnL_convolutive_noise → 多个陷波滤波串联，模拟线性/非线性通道带凹陷。"""
    x = sig.astype(np.float32)
    for _ in range(nBands):
        # 随机中心频率与带宽
        f0 = np.random.uniform(minF, min(maxF, fs/2 - 50.0))
        bw = np.random.uniform(minBW, maxBW)
        Q = f0 / max(1.0, bw)  # 品质因数
        try:
            b, a = iirnotch(w0=f0/(fs/2.0), Q=Q)
            x = filtfilt(b, a, x)
        except Exception:
            # 极端参数失败时跳过该陷波
            pass
    return x

def augment_isd(sig, fs, P=10, g_sd=2):
    """da2: ISD_additive_noise → 稀疏样本/片段加性噪声，占空比约 P%。"""
    x = sig.astype(np.float32).copy()
    N = len(x)
    # 以 P% 的概率对采样点注入短突发噪声
    num_impulses = int(N * P / 100.0 / 10) + 1
    for _ in range(num_impulses):
        pos = randint(0, max(0, N-1))
        span = randint(1, max(1, int(0.002 * fs)))  # ~2ms 短脉冲
        end = min(N, pos + span)
        noise = np.random.randn(end - pos).astype(np.float32) * g_sd * 1e-3
        x[pos:end] += noise
    return x

def _apply_colored_noise(x, fs, snr_db, color='pink'):
    """生成有色噪声并按 SNR 混合：支持 pink(1/f) / brown(1/f^2)/ white。"""
    N = len(x)
    # 生成白噪
    n = np.random.randn(N).astype(np.float32)
    if color in ('pink', 'brown'):
        # 频域着色
        X = rfft(n)
        freqs = np.linspace(1, N//2+1, len(X))  # 避免除0
        if color == 'pink':
            H = 1.0 / np.sqrt(freqs)
        else:  # brown
            H = 1.0 / (freqs)
        X_col = X * H
        n = irfft(X_col, n=N).astype(np.float32)
    # 调整功率实现目标 SNR
    Px = np.mean(x**2) + 1e-12
    Pn = np.mean(n**2) + 1e-12
    alpha = math.sqrt(Px / (Pn * (10**(snr_db/10.0))))
    n = n * alpha
    return x + n

def augment_ssi(sig, fs, SNRmin=10, SNRmax=40, color='pink'):
    """da3: SSI_additive_noise → 有色加性噪声，SNR ∈ [SNRmin, SNRmax] 随机。"""
    snr = np.random.uniform(SNRmin, SNRmax)
    return _apply_colored_noise(sig.astype(np.float32), fs, snr_db=snr, color=color)

def augment_da5_series(sig, fs,
                       nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
                       P=10, g_sd=2):
    """da5: series(1+2) → 先 LnL 陷波卷积，再 ISD 稀疏加噪。"""
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

def extract_lfcc(file, num_ceps=20, order_deltas=2, low_freq=0, high_freq=4000, augment=None):
    sig, fs = sf.read(file)
    # 可选：在特征前做波形增强
    if augment is not None:
        try:
            sig = augment(sig, fs)
        except Exception as e:
            logging.warning(f"Augment failed on {file}: {e}")
    lfccs = lfcc(sig=sig,
                 fs=fs,
                 num_ceps=num_ceps,
                 low_freq=low_freq,
                 high_freq=high_freq).T
    if order_deltas > 0:
        feats = [lfccs]
        for _ in range(order_deltas):
            feats.append(Deltas(feats[-1]))
        lfccs = vstack(feats)
    return lfccs

def extract_features(file, features, cached=False, augment=None):
    def get_feats():
        if features == 'lfcc':
            return extract_lfcc(file, augment=augment)
        else:
            return None
    # 有增强时不要写入通用缓存（避免不同随机增强污染同一键）
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
                #data.append(Tx.T)
                # 统一朝向：保证列数=特征维 d（而不是帧数 T）
                if len(data) == 0:
                # 以第一条确定 d_ref
                     F = Tx if Tx.shape[1] <= Tx.shape[0] else Tx.T   # 让列更可能是较小的“特征维”
                     d_ref = F.shape[1]
                else:
    # 对后续样本，找一个朝向使列数与 d_ref 一致；否则跳过该样本
                     if Tx.shape[1] == d_ref:
                        F = Tx
                     elif Tx.T.shape[1] == d_ref:
                        F = Tx.T
                     else:
        # 打个调试信息也行：print('skip init sample with shape', Tx.shape)
                        continue

                data.append(F)


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

    # EM training（此处也可选择对所有帧再跑 EM；如需完全与原版一致可保留 augment_fn）
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

                # E step
                weighted_log_prob = gmm._estimate_weighted_log_prob(Tx.T)
                log_prob_norm = logsumexp(weighted_log_prob, axis=1)
                with errstate(under='ignore'):
                    log_resp = weighted_log_prob - log_prob_norm[:, None]
                log_prob_norm_acc += log_prob_norm.sum()

                # M step accumulators
                resp = exp(log_resp)
                nk_acc += resp.sum(axis=0) + 10 * finfo(log(1).dtype).eps
                mu_acc += resp.T @ Tx.T
                sigma_acc += resp.T @ (Tx.T ** 2)

        # M step
        gmm.means_ = mu_acc / nk_acc[:, None]
        gmm.covariances_ = sigma_acc / nk_acc[:, None] - gmm.means_ ** 2 + gmm.reg_covar
        gmm.weights_ = nk_acc / n_samples
        gmm.weights_ /= gmm.weights_.sum()
        if (gmm.covariances_ <= 0.0).any():
            raise ValueError("ill-defined empirical covariance")
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
