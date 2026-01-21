import numpy as np
import sys
import os.path
import pandas
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

import matplotlib as mpl
from sklearn.metrics import roc_auc_score  # ✅ 新增：AUROC 计算


def plot_reliability_diagram_v2(labels, probs, n_bins=10, save_path='reliability_diagram.png', ece=None):
    """
    自定义可靠性图（绿色渐变 + 字体放大）
    横坐标: 平均置信度 (max(p, 1-p))
    纵坐标: 每个区间内的准确率
    颜色: 样本数量（浅绿 -> 深绿）
    """

    # 计算预测标签与置信度
    preds = (probs > 0.5).astype(int)
    confidence = np.where(probs > 0.5, probs, 1 - probs)
    correct = (preds == labels).astype(int)

    # 分 bin
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = np.logical_and(confidence >= bin_lower, confidence < bin_upper)
        bin_count = np.sum(in_bin)
        bin_counts.append(bin_count)
        if bin_count > 0:
            acc = np.mean(correct[in_bin])
            avg_conf = np.mean(confidence[in_bin])
        else:
            acc = np.nan
            avg_conf = np.nan
        accuracies.append(acc)
        bin_confidences.append(avg_conf)

    # 颜色映射: 浅绿 -> 深绿（按样本数量）
    cmap = plt.cm.Greens
    norm = mpl.colors.Normalize(vmin=min(bin_counts), vmax=max(bin_counts))
    colors = [cmap(norm(c)) for c in bin_counts]

    # 创建绘图对象
    fig, ax = plt.subplots(figsize=(8, 6))

    # 柱状图
    bar_width = 0.8 / n_bins
    bars = ax.bar(bin_centers, accuracies, width=bar_width, color=colors, edgecolor='black', linewidth=0.7)

    # 完美校准线
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5)

    # 样本数量标注
    for i, count in enumerate(bin_counts):
        if not np.isnan(accuracies[i]):
            ax.text(bin_centers[i], accuracies[i] + 0.03, f'{count}', ha='center', fontsize=12, fontweight='medium')
            
    # ECE 框（放在图上方）
    if ece is not None:
        ax.text(0.1, 0.8, f"ECE = {ece:.4f}", transform=ax.transAxes,
                fontsize=20, color='green', fontweight='bold')
            
    # 美化样式
    ax.set_xlabel('Confidence', fontsize=18)
    ax.set_ylabel('Accuracy', fontsize=18)
    ax.set_title('Reliability Diagram (Confidence vs Accuracy)', fontsize=18, pad=15, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=14)

    # ✅ 修复 colorbar：绑定到当前 figure
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 必须设置以防止警告
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('Sample Count', fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Reliability diagram saved to {save_path}")



def compute_ece_v2(labels, probs, n_bins=10):
    """
    计算 ECE (Expected Calibration Error)
    """
    preds = (probs > 0.5).astype(int)
    confidence = np.where(probs > 0.5, probs, 1 - probs)
    correct = (preds == labels).astype(int)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = np.logical_and(confidence >= bin_lower, confidence < bin_upper)
        if np.any(in_bin):
            acc = np.mean(correct[in_bin])
            avg_conf = np.mean(confidence[in_bin])
            ece += np.abs(acc - avg_conf) * np.sum(in_bin) / len(labels)
    return ece


def compute_calibration_metrics(cm_scores, invert=False, plot=True, output_prefix=""):
    """
    计算校准指标并绘制可靠性图
    - NLL
    - Brier Score
    - ECE（基于分类）
    - AUROC（基于分类对错的不确定性 AUROC，参考 GCN 脚本中的 compute_uncorr_auroc）
    """
    # 提取分数和标签
    raw_scores = cm_scores['1_x'].values
    labels = (cm_scores[5] == 'bonafide').astype(int).values  # bonafide=1, spoof=0    

    # 检查原始分数范围
    print(f"DEBUG: Raw scores range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
    
    # 如果分数不在[0,1]范围内，需要转换
    if raw_scores.min() < 0 or raw_scores.max() > 1:
        print("WARNING: Scores are not in [0,1] range, applying sigmoid transformation")
        probs = 1 / (1 + np.exp(-raw_scores))  # 概率 P(bonafide)
    else:
        probs = raw_scores
    
    print(f"DEBUG: Probabilities range: [{probs.min():.4f}, {probs.max():.4f}]")
    
    epsilon = 1e-15
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
    
    # ===== 1) NLL =====
    # binary cross-entropy: y in {0,1}, p = P(y=1)
    nll = -np.mean(labels * np.log(probs_clipped) + (1 - labels) * np.log(1 - probs_clipped))
    
    # ===== 2) Brier Score =====
    brier = np.mean((labels - probs) ** 2)
    
    # ===== 3) ECE（保持你原来的“基于分类”的实现）=====
    ece = compute_ece_v2(labels, probs, n_bins=10)

    # ===== 4) 基于分类对错的 AUROC（仿照 compute_uncorr_auroc 的思路）=====
    # 4.1 先做 hard classification：阈值 0.5 判 bonafide / spoof
    preds = (probs >= 0.5).astype(int)       # 预测标签 0/1
    correct = (preds == labels).astype(int)  # 对=1，错=0

    # 4.2 构造“正确 vs 错误”二分类标签：错=1，对=0
    y_binary = 1 - correct  # 错=1，对=0

    # 4.3 定义“不确定性得分”：这里用二分类熵，和多类版本的 -sum(p log p) 一致
    p = probs_clipped
    q = 1.0 - p
    uncertainty = -(p * np.log(p) + q * np.log(q))  # 熵，越大越不确定

    # 4.4 计算 AUROC（若全对或全错，则无法计算，返回 NaN）
    if (y_binary.sum() == 0) or (y_binary.sum() == y_binary.size):
        print("WARNING: All predictions are either correct or wrong only; cannot compute uncorr AUROC.")
        auroc = float('nan')
    else:
        auroc = roc_auc_score(y_binary, uncertainty)

    # ===== 5) 绘制可靠性图（仍然基于分类结果的置信度）=====
    save_path = f"{output_prefix}_reliability_diagram.png"
    plot_reliability_diagram_v2(labels, probs, n_bins=10, save_path=save_path, ece=ece)

    # 打印（注意处理 NaN）
    if np.isnan(auroc):
        auroc_str = "nan"
    else:
        auroc_str = f"{auroc:.4f}"

    print(f"NLL={nll:.4f}, Brier={brier:.4f}, ECE={ece:.4f}, AUROC(uncorr)={auroc_str}")
    
    # 调试信息
    print(f"DEBUG: Label distribution: {np.bincount(labels)} (0=spoof, 1=bonafide)")
    print(f"DEBUG: Mean probability: {probs.mean():.4f}")
    
    return nll, brier, ece, auroc



def compute_det_curve(target_scores, nontarget_scores):
    # 添加空数组检查
    if target_scores.size == 0 or nontarget_scores.size == 0:
        print("WARNING: target_scores or nontarget_scores is empty")
        # 返回默认值
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([-np.inf, np.inf])

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    # 添加空数组检查
    if target_scores.size == 0 or nontarget_scores.size == 0:
        print("WARNING: Cannot compute EER - target_scores or nontarget_scores is empty")
        return 0.5, 0.0  # 返回默认值
    
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]
    
def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv

def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, print_cost):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.
    """
    # Sanity check of cost parameters
    if cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0 or \
            cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pfa_spoof_asv is None:
        sys.exit('ERROR: you should provide false alarm rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan

    C0 = cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon']*cost_model['Cfa']*Pfa_asv
    C1 = cost_model['Ptar'] * cost_model['Cmiss'] - (cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv)
    C2 = cost_model['Pspoof'] * cost_model['Cfa_spoof'] * Pfa_spoof_asv;


    # Sanity check of the weights
    if C0 < 0 or C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm

    # Obtain default t-DCF
    tDCF_default = C0 + np.minimum(C1, C2)

    # Normalized t-DCF
    tDCF_norm = tDCF / tDCF_default

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa          = {:8.5f} (Cost of tandem system falsely accepting a nontarget)'.format(cost_model['Cfa']))
        print('   Cmiss        = {:8.5f} (Cost of tandem system falsely rejecting target speaker)'.format(cost_model['Cmiss']))
        print('   Cfa_spoof    = {:8.5f} (Cost of tandem sysmte falsely accepting spoof)'.format(cost_model['Cfa_spoof']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), t_CM=CM threshold)')
        print('   tDCF_norm(t_CM) = {:8.5f} + {:8.5f} x Pmiss_cm(t_CM) + {:8.5f} x Pfa_cm(t_CM)\n'.format(C0/tDCF_default, C1/tDCF_default, C2/tDCF_default))
        print('     * The optimum value is given by the first term (0.06273). This is the normalized t-DCF obtained with an error-free CM system.')
        print('     * The minimum normalized cost (minimum over all possible thresholds) is always <= 1.00.')
        print('')

    return tDCF_norm, CM_thresholds

def compute_tDCF_legacy(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost):
    """
    Legacy t-DCF 计算（保留原版接口，未修改）
    """
    # Sanity check of cost parameters
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: you should provide miss rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'.format(cost_model['Cfa_asv']))
        print('   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'.format(cost_model['Cmiss_asv']))
        print('   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'.format(cost_model['Cfa_cm']))
        print('   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'.format(cost_model['Cmiss_cm']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)')

        if C2 == np.minimum(C1, C2):
            print('   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(C1 / C2))
        else:
            print('   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(C2 / C1))

    return tDCF_norm, CM_thresholds

#!/usr/bin/env python
"""
Script to compute pooled EER and min tDCF for ASVspoof2021 LA. 
"""

from glob import glob

truth_dir = '/root/autodl-tmp/datasets/ASVspoof2021_LA_eval/'

# 新增：读取eval协议文件
eval_protocol_file = '/root/autodl-tmp/datasets/ASVspoof2021_LA_eval/custom_protocols_21LA/ASVspoof2021_LA/asvspoof2021_la.eval.trl.txt'

def load_eval_trials(protocol_file):
    """从eval协议文件中加载所有trial ID"""
    eval_trials = set()
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                # 协议文件格式: "- LA_E_2435789 - - spoof"
                # 第二列是trial ID
                eval_trials.add(parts[1])
    return eval_trials

asv_key_file = os.path.join(truth_dir, 'ASV_trial_metadata.txt')
asv_scr_file = os.path.join(truth_dir, 'score.txt')
cm_key_file = os.path.join(truth_dir, 'CM_trial_metadata.txt')

Pspoof = 0.05
cost_model = {
    'Pspoof': Pspoof,  # Prior probability of a spoofing attack
    'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker
    'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker
    'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof,
}

def load_asv_metrics(phase):
    """加载ASV指标，但只针对eval trials"""
    # 加载eval trials
    eval_trials = load_eval_trials(eval_protocol_file)
    print(f"DEBUG: Loaded {len(eval_trials)} trials from protocol file")
    
    # 加载组织者的ASV分数
    asv_key_data = pandas.read_csv(asv_key_file, sep=' ', header=None)
    asv_scr_data = pandas.read_csv(asv_scr_file, sep=' ', header=None)
    
    print(f"DEBUG: Total ASV key data rows: {len(asv_key_data)}")
    print(f"DEBUG: Total ASV score data rows: {len(asv_scr_data)}")
    
    # 检查phase列的唯一值
    unique_phases = asv_key_data[7].unique()
    print(f"DEBUG: Unique phases in ASV key data: {unique_phases}")
    
    # 筛选出属于指定phase的数据
    phase_mask = asv_key_data[7] == phase
    print(f"DEBUG: Rows matching phase '{phase}': {phase_mask.sum()}")
    
    # ASV文件第二列格式: "LA_E_5013670-alaw-ita_tx"
    # 提取基础ID部分进行匹配
    asv_base_ids = asv_key_data[1].str.extract(r'(LA_[A-Z]_\d+)', expand=False)
    print(f"DEBUG: Sample extracted base IDs: {asv_base_ids.head(3).tolist()}")
    eval_mask = asv_base_ids.isin(eval_trials)
    print(f"DEBUG: Rows matching eval trials (base IDs): {eval_mask.sum()}")
    
    combined_mask = phase_mask & eval_mask
    print(f"DEBUG: Combined mask (phase & eval): {combined_mask.sum()}")
    
    # 应用筛选
    asv_key_filtered = asv_key_data[combined_mask]
    asv_scr_filtered = asv_scr_data[combined_mask]
    
    print(f"DEBUG: Found {len(asv_key_filtered)} ASV trials for phase '{phase}'")
    
    # 提取目标、非目标和欺骗分数
    idx_tar = asv_key_filtered[5] == 'target'
    idx_non = asv_key_filtered[5] == 'nontarget'
    idx_spoof = asv_key_filtered[5] == 'spoof'

    tar_asv = asv_scr_filtered[2][idx_tar]
    non_asv = asv_scr_filtered[2][idx_non]
    spoof_asv = asv_scr_filtered[2][idx_spoof]
    
    print(f"DEBUG: target samples: {len(tar_asv)}, non-target samples: {len(non_asv)}, spoof samples: {len(spoof_asv)}")
    
    # 检查是否有足够的样本计算EER
    if len(tar_asv) == 0 or len(non_asv) == 0:
        print(f"ERROR: Cannot compute ASV EER - insufficient samples (target: {len(tar_asv)}, non-target: {len(non_asv)})")
        print("WARNING: Using default ASV error rates due to insufficient samples")
        return 0.1, 0.1, 0.1, 0.1
    
    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold
    )

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv
    
def performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=False):
    bona_cm = cm_scores[cm_scores[5]=='bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5]=='spoof']['1_x'].values

    if invert==False:
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    else:
        eer_cm = compute_eer(-bona_cm, -spoof_cm)[0]

    if invert==False:
        tDCF_curve, _ = compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)
    else:
        tDCF_curve, _ = compute_tDCF(-bona_cm, -spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    return min_tDCF, eer_cm

def eval_to_score_file(score_file, cm_key_file, phase):
    """评估分数文件，但只针对eval trials"""
    # 对于hidden track，跳过ASV指标计算或使用默认值
    if phase == "hidden":
        print("Skipping ASV metrics for hidden track")
        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = 0.1, 0.1, 0.1, 0.1  # 使用默认值
    else:
        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics(phase)
    
    # 加载eval trials
    eval_trials = load_eval_trials(eval_protocol_file)
    
    # 加载CM元数据
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)

    # CM文件第二列格式: "LA_E_9332881" (没有后缀)
    # 直接匹配
    cm_data_eval = cm_data[cm_data[1].isin(eval_trials)]
    
    if len(submission_scores) != len(cm_data_eval):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data_eval)))
        print('Expected trials in eval set:', len(cm_data_eval))
        print('Found in submission:', len(submission_scores))
        exit(1)

    # 合并分数和元数据 - 只合并eval trials
    cm_scores = submission_scores.merge(cm_data_eval[cm_data_eval[7] == phase], left_on=0, right_on=1, how='inner')
    
    print(f"Evaluating {len(cm_scores)} trials from {phase} set")
    
    min_tDCF, eer_cm = performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv)

    out_data = "eer: %.2f\n" % (100*eer_cm)
    out_data += "min_tDCF: %.4f\n" % min_tDCF
    print(out_data, end="")

    # 计算校准指标 + AUROC
    print("\n=== Calibration Metrics ===")
    nll, brier, ece, auroc = compute_calibration_metrics(
        cm_scores, 
        invert=False, 
        output_prefix=os.path.splitext(score_file)[0]
    )
    
    # 保存校准指标
    cal_metrics_file = os.path.splitext(score_file)[0] + "_calibration_metrics.txt"
    with open(cal_metrics_file, 'w') as f:
        f.write(f"NLL: {nll:.6f}\n")
        f.write(f"Brier Score: {brier:.6f}\n")
        f.write(f"ECE: {ece:.6f}\n")
        f.write(f"AUROC: {auroc:.6f}\n")
    
    print(f"Calibration metrics saved to {cal_metrics_file}")
    
    # 检查分数符号
    min_tDCF2, eer_cm2 = performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=True)

    if min_tDCF2 < min_tDCF:
        print(f'CHECK: we negated your scores and achieved a lower min t-DCF. Before: {min_tDCF:.3f} - Negated: {min_tDCF2:.3f}')

    return min_tDCF

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scoreFile', type=str, default="")
    parser.add_argument('--phase', type=str, default="eval")
    args = parser.parse_args()

    def remove_duplicate_lines_inplace(file_path):
        seen_first_strings = set()
        with open(file_path, 'r+') as file:
            lines = file.readlines()
            file.seek(0)
            file.truncate()
            for line in lines:
                first_string = line.split()[0]
                if first_string not in seen_first_strings:
                    file.write(line)
                    seen_first_strings.add(first_string)

    remove_duplicate_lines_inplace(args.scoreFile)
    eval_to_score_file(args.scoreFile, cm_key_file, args.phase)
