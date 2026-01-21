import numpy as np
import sys
import os.path
import pandas
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score


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
    vmin, vmax = min(bin_counts), max(bin_counts)
    if vmin == vmax:
        vmax = vmin + 1e-6
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
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
            y_text = min(accuracies[i] + 0.03, 0.97)
            ax.text(bin_centers[i], y_text, f'{count}', ha='center', fontsize=12, fontweight='medium')

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

    # colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
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
    - AUROC（基于“预测对错 vs 不确定性”的 uncorr AUROC）
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
    nll = -np.mean(labels * np.log(probs_clipped) + (1 - labels) * np.log(1 - probs_clipped))

    # ===== 2) Brier Score =====
    brier = np.mean((labels - probs) ** 2)

    # ===== 3) ECE =====
    ece = compute_ece_v2(labels, probs, n_bins=10)

    # ===== 4) uncorr AUROC（和你 LA 脚本逻辑一致）=====
    preds = (probs >= 0.5).astype(int)       # 预测标签 0/1
    correct = (preds == labels).astype(int)  # 对=1，错=0

    # 错=1，对=0
    y_binary = 1 - correct

    # 二分类熵作为“不确定性”
    p = probs_clipped
    q = 1.0 - p
    uncertainty = -(p * np.log(p) + q * np.log(q))

    if (y_binary.sum() == 0) or (y_binary.sum() == y_binary.size):
        print("WARNING: All predictions are either correct or wrong only; cannot compute uncorr AUROC.")
        auroc = float('nan')
    else:
        auroc = roc_auc_score(y_binary, uncertainty)

    # ===== 5) 绘图 =====
    if plot:
        save_path = f"{output_prefix}_reliability_diagram.png"
        plot_reliability_diagram_v2(labels, probs, n_bins=10, save_path=save_path, ece=ece)

    # 打印
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
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds

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


def load_eval_trials(protocol_file):
    """从eval协议文件中加载所有trial ID和标签"""
    eval_trials = {}
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                trial_id = parts[1]
                label = parts[-1]  # 第4列是标签
                eval_trials[trial_id] = label
    return eval_trials


def eval_to_score_file(score_file, cm_key_file, protocol_file, phase):
    # 加载eval trials
    eval_trials = load_eval_trials(protocol_file)

    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)

    # 筛选出属于eval trials的数据
    cm_data_eval = cm_data[cm_data[1].isin(eval_trials.keys())]

    if len(submission_scores) != len(cm_data_eval):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data_eval)))
        print('Expected trials in eval set:', len(cm_data_eval))
        print('Found in submission:', len(submission_scores))
        exit(1)

    # 合并分数和元数据 - 只合并eval trials
    cm_scores = submission_scores.merge(cm_data_eval[cm_data_eval[7] == phase],
                                        left_on=0, right_on=1, how='inner')

    print(f"Evaluating {len(cm_scores)} trials from eval set")

    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    out_data = "eer: %.2f\n" % (100 * eer_cm)
    print(out_data)

    # 计算校准指标并绘制可靠性图 + AUROC
    print("\n=== Calibration Metrics ===")
    nll, brier, ece, auroc = compute_calibration_metrics(
        cm_scores,
        invert=False,
        plot=True,
        output_prefix=os.path.splitext(score_file)[0]
    )

    # 保存校准指标到文件
    cal_metrics_file = os.path.splitext(score_file)[0] + "_calibration_metrics.txt"
    with open(cal_metrics_file, 'w') as f:
        f.write(f"EER: {100 * eer_cm:.2f}%\n")
        f.write(f"NLL: {nll:.6f}\n")
        f.write(f"Brier Score: {brier:.6f}\n")
        f.write(f"ECE: {ece:.6f}\n")
        if np.isnan(auroc):
            f.write(f"AUROC(uncorr): nan\n")
        else:
            f.write(f"AUROC(uncorr): {auroc:.6f}\n")

    print(f"Calibration metrics saved to {cal_metrics_file}")

    return eer_cm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scoreFile', type=str, default="")
    parser.add_argument('--phase', type=str, default="eval")
    parser.add_argument('--protocol_file', type=str,
                        default='/root/autodl-tmp/datasets/ASVspoof2021_DF_eval/custom_protocols_21DF/ASVspoof2021_DF/asvspoof2021_df.eval.trl.txt')
    args = parser.parse_args()

    cm_key_file = "/root/autodl-tmp/datasets/ASVspoof2021_DF_eval/trial_metadata.txt"

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
    eval_to_score_file(args.scoreFile, cm_key_file, args.protocol_file, args.phase)
