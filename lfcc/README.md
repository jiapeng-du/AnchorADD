# Running Instructions

This repository provides baseline and uncertainty-aware (UQ-enabled) GMM-based anti-spoofing experiments on multiple ASVspoof datasets. For each dataset, the experiment follows a unified three-stage pipeline:

1. **Model training**
2. **Model evaluation / scoring**
3. **Metric computation from score files**

Below, we describe the complete procedure using **ASVspoof 2021 LA** as an example. Other datasets (e.g., ASVspoof2019 LA, ASVspoof2021 DF, In-the-Wild) follow the same workflow with dataset-specific scripts and protocol files.

---

## 1. Training (ASVspoof 2021 LA)

Two training modes are supported: **without UQ** (baseline) and **with UQ** (ΔUQ-enabled).

### 1.1 Baseline Training (without UQ)

Run the baseline GMM training script:

```bash
python asvspoof2021_baseline.py
```

This script:

* Extracts standard LFCC features (no uncertainty modeling)
* Trains separate GMMs for *bonafide* and *spoof* speech
* Saves the trained GMM parameters to a model file for later evaluation

---

### 1.2 UQ-Enabled Training (with ΔUQ)

Run the uncertainty-aware training script:

```bash
python asvspoof2021_baseline_gduq.py
```

In this mode:

* ΔUQ-enhanced LFCC features are used
* The training pipeline remains identical to the baseline setting
* A separate GMM parameter file is produced for the UQ-enabled model

---

## 2. Evaluation / Scoring (ASVspoof 2021 LA)

After training, the trained GMM models are evaluated on the target protocol (e.g., **dev** or **eval**) to generate score files.

### 2.1 Baseline Scoring (without UQ)

Run:

```bash
python gmm_scoring_asvspoof21.py
```

This script:

* Loads the baseline GMM parameters
* Performs GMM scoring on the evaluation set
* Outputs a score file (e.g., `scores-*.txt`)

---

### 2.2 UQ-Enabled Scoring (with ΔUQ)

Run:

```bash
python gmm_scoring_asvspoof21_gduq.py
```

This script:

* Loads the UQ-enabled GMM parameters
* Uses ΔUQ features consistently with the training stage
* Produces the corresponding UQ-aware score file

---

## 3. Metric Computation

Once the score files are generated, performance metrics are computed using scripts in the `tools/` directory.

For example:

```bash
python tools/cul_eer_final.py --scoreFile path/to/your_score_file.txt
```

This step computes standard anti-spoofing evaluation metrics such as:

* Equal Error Rate (EER)
* Other metrics (e.g., t-DCF), depending on the specific evaluation script

---

## 4. Other Datasets

Experiments on other datasets (e.g., **ASVspoof2019 LA**, **ASVspoof2021 DF**, **In-the-Wild**) follow the same structure:

* **Training**:

  * `*_baseline.py` → baseline (without UQ)
  * `*_baseline_gduq.py` → UQ-enabled
* **Scoring**:

  * `gmm_scoring_*.py` → baseline scoring
  * `gmm_scoring_*_gduq.py` → UQ-enabled scoring
* **Evaluation**:

  * Run the corresponding script in `tools/` on the generated score files

Only dataset paths and protocol files differ across datasets.

---

## 5. Overall Workflow Summary

```text
Training (no UQ / with UQ)
        ↓
Evaluation / Scoring
        ↓
Score Files (*.txt)
        ↓
Metric Computation (EER, etc.)
```

This unified pipeline enables fair and consistent comparison between baseline and UQ-enabled models across all datasets.
