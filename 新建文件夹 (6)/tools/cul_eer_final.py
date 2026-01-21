import numpy as np
import argparse
import sys
import matplotlib
# 使用非交互式后端，适用于服务器环境
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path

from sklearn.metrics import roc_auc_score  # ✅ 新增：用于计算 uncorr-AUROC


#def plot_reliability_diagram(labels, probs, n_bins=15, save_path='reliability_diagram.png', ece_value=None): # 改为个bins
#    print("Example (probs, labels):")
#    for i in range(10):
#        print(f"{probs[i]}, {labels[i]}")
#
#    bin_boundaries = np.linspace(0, 1, n_bins + 1)
#    bin_lowers = bin_boundaries[:-1]
#    bin_uppers = bin_boundaries[1:]
#    bin_centers = (bin_lowers + bin_uppers) / 2
#
#    accuracies = []
#    confidences = []
#    counts = []
#
#    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
#        in_bin = np.logical_and(probs >= bin_lower, probs < bin_upper)
#        if np.any(in_bin):
#            prob_in_bin = probs[in_bin]
#            label_in_bin = labels[in_bin]
#            avg_confidence = np.mean(prob_in_bin)
#            avg_accuracy = np.mean(label_in_bin)
#            count = np.sum(in_bin)
#
#            confidences.append(avg_confidence)
#            accuracies.append(avg_accuracy)
#            counts.append(count)
#        else:
#            # 处理空桶
#            confidences.append(np.nan)
#            accuracies.append(np.nan)
#            counts.append(0)
#            
#    print("Bin summary:")
#    for i, (center, acc, count) in enumerate(zip(bin_centers, accuracies, counts)):
#        print(f"Bin {i}: center={center:.2f}, acc={acc}, n={count}")
#
#    # 开始绘图
#    plt.figure(figsize=(8, 8))
#    # 1. 绘制完美校准对角线
#    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=2)
#
#    # 2. 绘制校准曲线（条形图）
#    width = 0.8 / n_bins # 条形宽度
#    plt.bar(bin_centers, accuracies, width=width, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
#
#    # 3. 可选：在每个条形上方标注样本数量
#    for i, (center, acc, count) in enumerate(zip(bin_centers, accuracies, counts)):
#        if count > 0 and not np.isnan(acc):
#            plt.annotate(f'n={count}', xy=(center, acc), xytext=(0, 5),
#                         textcoords='offset points', ha='center', fontsize=8)
#    # 添加ECE值标注
#    if ece_value is not None:
#        plt.text(0.1, 0.9, f'ECE = {ece_value:.4f}', transform=plt.gca().transAxes,
#                 bbox=dict(facecolor='white', alpha=0.7))
#
#    plt.xlabel('Predicted Probability (Confidence Bin)')
#    plt.ylabel('Actual Accuracy')
#    plt.title('Reliability Diagram')
#    plt.legend(loc='upper left')
#    plt.grid(True, alpha=0.3)
#    plt.xlim(0, 1)
#    plt.ylim(0, 1)
#    plt.tight_layout()
#    plt.savefig(save_path, dpi=300, bbox_inches='tight')
#    plt.close()
#    print(f"Reliability diagram (bar style) saved to {save_path}")
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl


def plot_reliability_diagram_v2(labels, probs, n_bins=10, save_path='reliability_diagram.png', ece=None):
    """
    自定义可靠性图（绿色渐变 + 字体放大）
    横坐标: 平均置信度 (max(p, 1-p))
    纵坐标: 每个区间内的准确率
    颜色: 样本数量（浅绿 -> 深绿）
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

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
    ax.text(0.1, 0.8, f"ECE = {ece:.4f}", transform=ax.transAxes,fontsize=20,color='green',fontweight='bold')
#    plt.text(0.1, 0.9, f'ECE = {ece:.4f}', transform=plt.gca().transAxes,
#                 bbox=dict(facecolor='white', alpha=0.7))            
    # 美化样式
    ax.set_xlabel('Confidence', fontsize=18)
    ax.set_ylabel('Accuracy', fontsize=18)
    ax.set_title('Reliability Diagram (Confidence vs Accuracy)', fontsize=18, pad=15, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
#    ax.legend(fontsize=12, loc='upper left')
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


def compute_nll_brier_ece_v2(score_file, label_file, pos=1):
    """
    读取分数文件与标签文件，并计算指标 + 绘图
    现在额外增加：uncorr-AUROC（基于分类对错 + 熵不确定性的 AUROC）
    """
    # 读取真实标签
    true_labels_dict = {}
    with open(label_file, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 4:
                wav_id, label = parts[1], parts[4]
                true_labels_dict[wav_id] = 0 if label == "spoof" else 1

    # 读取预测分数
    raw_scores, labels = [], []
    with open(score_file, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > pos:
                wav_id, score = parts[0], float(parts[pos])
                if wav_id in true_labels_dict:
                    raw_scores.append(score)
                    labels.append(true_labels_dict[wav_id])

    raw_scores = np.array(raw_scores)
    labels = np.array(labels)

    # ������ 修复：检查并转换分数到概率范围
    print(f"DEBUG: Raw scores range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
    
    # 如果分数不在[0,1]范围内，需要转换
    if raw_scores.min() < 0 or raw_scores.max() > 1:
        print("WARNING: Scores are not in [0,1] range, applying sigmoid transformation")
        # 使用sigmoid转换到[0,1]
        probs = 1 / (1 + np.exp(-raw_scores))
    else:
        probs = raw_scores
    
    print(f"DEBUG: Probabilities range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"DEBUG: Label distribution: {np.bincount(labels)} (0=spoof, 1=bonafide)")

    # 计算指标
    epsilon = 1e-15
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
    
    # 正确计算NLL
    nll = -np.mean(labels * np.log(probs_clipped) + (1 - labels) * np.log(1 - probs_clipped))
    
    # 正确计算Brier Score
    brier = np.mean((labels - probs) ** 2)
    
    # ✅ 保持原来的ECE计算方式
    ece = compute_ece_v2(labels, probs, n_bins=10)

    # ✅ 新增：uncorr-AUROC（基于分类对错的不确定性 AUROC）
    # 1) 先做 hard classification（阈值 0.5）
    preds = (probs >= 0.5).astype(int)
    correct = (preds == labels).astype(int)  # 对=1，错=0

    # 2) 构造“错 vs 对”的二分类标签：错=1，对=0
    y_binary = 1 - correct

    # 3) 定义不确定性：用二分类熵 -[p log p + (1-p) log (1-p)]
    p = probs_clipped
    q = 1.0 - p
    uncertainty = -(p * np.log(p) + q * np.log(q))

    # 4) 计算 AUROC，如果全对或全错则无法算，返回 NaN
    if (y_binary.sum() == 0) or (y_binary.sum() == y_binary.size):
        print("WARNING: All predictions are either correct or wrong only; cannot compute uncorr-AUROC.")
        auroc_uncorr = float('nan')
    else:
        auroc_uncorr = roc_auc_score(y_binary, uncertainty)

    # 绘图
    save_path = f"{os.path.splitext(score_file)[0]}_reliability_diagram_v2.png"
    plot_reliability_diagram_v2(labels, probs, n_bins=10, save_path=save_path, ece=ece)

    # 打印指标
    if np.isnan(auroc_uncorr):
        auroc_str = "nan"
    else:
        auroc_str = f"{auroc_uncorr:.4f}"

    print(f"NLL={nll:.4f}, Brier={brier:.4f}, ECE={ece:.4f}, Uncorr-AUROC={auroc_str}")
    
    # 额外的调试信息
    bonafide_probs = probs[labels == 1]
    spoof_probs = probs[labels == 0]
    print(f"DEBUG: Bonafide mean prob: {np.mean(bonafide_probs):.4f}")
    print(f"DEBUG: Spoof mean prob: {np.mean(spoof_probs):.4f}")
    
    return nll, brier, ece, auroc_uncorr


def eer19only(score_file, label_file,pos=1):
    target=[]
    nontarget=[]
    target_score=[]
    nontarget_score=[]
    wav_lists=[]
    score={}
    lable_list={}
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                label=line[4]
                if label=="spoof":
                    nontarget.append(wav_id)
                else:
                    target.append(wav_id)

    with open(score_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[0]
                score[wav_id]=(line[pos]).replace("[","").replace("]","")
    for wav_id in target:
        target_score.append(float(score[wav_id]))
    for wav_id in nontarget:
        nontarget_score.append(float(score[wav_id]))
    target_score=np.array(target_score)
    nontarget_score=np.array(nontarget_score)
    eer_cm, _=compute_eer(target_score, nontarget_score)
    return eer_cm * 100



def eeronly(score_file, label_file,pos=1):
    target=[]
    nontarget=[]
    target_score=[]
    nontarget_score=[]
    wav_lists=[]
    score={}
    lable_list={}
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[2]
                label=line[5]
                if label=="deepfake":
                    nontarget.append(wav_id)
                else:
                    target.append(wav_id)

    with open(score_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[0]
                score[wav_id]=(line[pos]).replace("[","").replace("]","")
    for wav_id in target:
        target_score.append(float(score[wav_id]))
    for wav_id in nontarget:
        nontarget_score.append(float(score[wav_id]))
    target_score=np.array(target_score)
    nontarget_score=np.array(nontarget_score)
    eer_cm, _=compute_eer(target_score, nontarget_score)
    return eer_cm * 100



def get_alltrn_data_kv():
    # 初始化一个空字典
    key_value_dict = {}

    # 打开txt文件进行读取
    with open('/root/autodl-tmp/datasets/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt', 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 使用空格分割每行内容，并取第二列作为key
            key = line.split()[1]
            # 初始化value为-1
            value = 0
            # 将key-value对添加到字典中
            key_value_dict[key] = value
    return key_value_dict


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):

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
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost):
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
    INPUTS:
      bonafide_score_cm   A vector of POSITIVE CPASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CPASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.
                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss_asv   Cost of ASV falsely rejecting target.
                          Cfa_asv     Cost of ASV falsely accepting nontarget.
                          Cmiss_cm    Cost of CM falsely rejecting target.
                          Cfa_cm      Cost of CM falsely accepting spoof.
      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?
    OUTPUTS:
      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).
    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.
    References:
      [1] T. Kinnunen, K.-A. Lee, H. Delgado, N. Evans, M. Todisco,
          M. Sahidullah, J. Yamagishi, D.A. Reynolds: "t-DCF: a Detection
          Cost Function for the Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification", Proc. Odyssey 2018: the
          Speaker and Language Recognition Workshop, pp. 312--319, Les Sables d'Olonne,
          France, June 2018 (https://www.isca-speech.org/archive/Odyssey_2018/pdfs/68.pdf)
      [2] ASVspoof 2019 challenge evaluation plan
          TODO: <add link>
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

def eerandtdcf(score_file, label_file, asv_label,pos=1):

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }
    asv_data = np.genfromtxt(asv_label, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float64)

    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    target=[]
    nontarget=[]
    target_score=[]
    nontarget_score=[]
    wav_lists=[]
    score={}
    lable_list={}
    wrong=0
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                label=line[4]
                lable_list[wav_id]=label
                if label=="spoof":
                    nontarget.append(wav_id)
                else:
                    target.append(wav_id)

    with open(score_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[0]
                wav_lists.append(wav_id)
                score[wav_id]=(line[pos]).replace("[","").replace("]","")
    for wav_id in target:
        target_score.append(float(score[wav_id]))
    for wav_id in nontarget:
        nontarget_score.append(float(score[wav_id]))
    target_score=np.array(target_score)
    nontarget_score=np.array(nontarget_score)
    eer_cm, Threshhold=compute_eer(target_score, nontarget_score)
    '''
    print("EER={}, Threshhold={}".format(EER, Threshhold))
    for wav_id in wav_lists:
        if float(score[wav_id])>Threshhold and lable_list[wav_id]=="spoof":
            wrong+=1
    
    acc=(len(score)-wrong)/len(score)
    print("Acc={}".format(acc))
    '''
    tDCF_curve, CM_thresholds = compute_tDCF(target_score, nontarget_score, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    print('ASV SYSTEM')
    print('   EER            = {:8.5f} % (Equal error rate (target vs. nontarget discrimination)'.format(eer_asv * 100))
    print('   Pfa            = {:8.5f} % (False acceptance rate of nontargets)'.format(Pfa_asv * 100))
    print('   Pmiss          = {:8.5f} % (False rejection rate of targets)'.format(Pmiss_asv * 100))
    print('   1-Pmiss,spoof  = {:8.5f} % (Spoof false acceptance rate)'.format((1 - Pmiss_spoof_asv) * 100))

    print('\nCM SYSTEM')
    print('   EER            = {:8.5f} % (Equal error rate for countermeasure)'.format(eer_cm * 100))

    print('\nTANDEM')
    print('   min-tDCF       = {:8.5f}'.format(min_tDCF))
    
    return eer_cm * 100, min_tDCF


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset
    # parser.add_argument('--layer', type=int, default=0)
    # parser.add_argument('--type', type=str, default="")
    parser.add_argument('--scoreFile', type=str, default="")
    parser.add_argument('--pos', type=int, default=1) #列数-1
    args = parser.parse_args()
    
    # partial_spoof
    # labelFile="/data8/wangzhiyong/project/fakeAudioDetection/vocoderReWavFAD/datasets/partial_spoof/protocol/PartialSpoof.LA.cm.eval.trl.txt"
    # asvlabel="/data8/wangzhiyong/project/fakeAudioDetection/vocoderReWavFAD/datasets/partial_spoof/protocol/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    
    # asvspoof 2019
    labelFile="/root/autodl-tmp/datasets/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    asvlabel="/root/autodl-tmp/datasets/ASVspoof2019_LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"

    # inthewild
    # labelFile="/data8/wangzhiyong/project/fakeAudioDetection/FAD_research/datasets/release_in_the_wild/inthewild_protocol.txt"

    # scoreFile=f"{args.type}log_eval_partialspoof_{args.layer}_score.txt"
    
    


    def remove_duplicate_lines_inplace(file_path):
        # 用于存储已经遇到的第一个字符串
        seen_first_strings = set()

        # 打开文件进行读写
        with open(file_path, 'r+') as file:
            lines = file.readlines()  # 读取所有行

            # 将文件指针移到文件开头，准备写入新的内容
            file.seek(0)
            file.truncate()  # 清空文件内容

            for line in lines:
                # 提取每行的第一个字符串
                first_string = line.split()[0]

                # 如果第一个字符串没有重复，写入文件并添加到集合中
                if first_string not in seen_first_strings:
                    file.write(line)
                    seen_first_strings.add(first_string)

    # 调用函数，传入文件路径
    remove_duplicate_lines_inplace(args.scoreFile)
    
    
    eerandtdcf(args.scoreFile, labelFile, asvlabel,pos=args.pos)
    nll, brier, ece, auroc_uncorr = compute_nll_brier_ece_v2(args.scoreFile, labelFile, pos=args.pos)
#    if nll is not None:
#           print(f"NLL: {nll:.6f}")
#           print(f"Brier Score: {brier:.6f}")
#           print(f"ECE: {ece:.6f}")
