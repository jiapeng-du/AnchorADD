# MoE Experiment Running Instructions

This document describes how to run the **Mixture-of-Experts (MoE)** experiments in this repository, including **training**, **inference (testing)**, and **evaluation**. The workflow is illustrated using **ASVspoof2019 LA (19LA)** as an example. Other datasets (**LA21, DF21, ITW**) follow the same procedure with different dataset flags.

---

## 0. Prerequisites: Dataset Path Configuration

Before running any experiment, make sure all dataset paths and protocol files are correctly configured.

Edit the following file:

```text
utils/loadData/asvspoof_data_DA2.py
```

Check and update:

* Protocol file paths (train / dev / eval)
* Audio file root directories

⚠️ This step is **required** before both training and inference.

---

## 1. Training (MoE)

### Example: ASVspoof2019 LA (19LA)

### Step 1: Select Dataset

Open `main2.py` and set the dataset identifier:

```python
datasets_name = 19  # ASVspoof2019 LA
```

Other supported options:

* `LA21` → ASVspoof2021 LA
* `DF21` → ASVspoof2021 DF
* `ITW`  → In-the-Wild

---

### Step 2: Configure Data Augmentation for Training

Open:

```text
utils/arg_parse.py
```

Modify **line 359**:

```python
algo = 5
```

This enables **DA5**, which is used during the **training stage**.

---

### Step 3: Run Training

Execute the training command:

```bash
python main2.py --gpuid 0 --save_dir <your_output_directory>
```

Arguments:

* `--gpuid 0` : GPU index
* `--save_dir` : directory for saving experiment outputs

After training finishes, the output directory will contain:

* Trained MoE model checkpoints
* Experiment logs
* Versioned folders (e.g., `version_0/`)

---

## 2. Inference / Testing (MoE)

Inference is performed using a trained checkpoint obtained from the training stage.

### Step 1: Configure Data Augmentation for Inference

Edit `utils/arg_parse.py` again and change **line 359** to:

```python
algo = 3  # or algo = 7
```

* `algo = 3` or `algo = 7` corresponds to **DA3 / DA7**
* These settings are used **only for inference/testing**

---

### Step 2: Run Inference

Run the following command:

```bash
python main2.py \
  --inference \
  --trained_model <path_to_trained_checkpoint> \
  --testset 19
```

Example:

```bash
python main2.py \
  --inference \
  --trained_model ./2/DF21_uq/version_0 \
  --testset 19
```

Arguments:

* `--trained_model` : path to the saved training directory
* `--testset` : dataset used for evaluation

Test set options:

* `19`   → ASVspoof2019 LA
* `LA21` → ASVspoof2021 LA
* `DF21` → ASVspoof2021 DF
* `ITW`  → In-the-Wild

---

### Step 3: Inference Output

After inference, a score file will be generated, for example:

```text
infer_19.log
```

This file contains system scores (and uncertainty-related outputs if enabled) for all evaluation trials.

---

## 3. Evaluation and Uncertainty Metrics

Final evaluation is performed using scripts in:

```text
utils/tools/
```

Use the dataset-specific evaluation script:

### ASVspoof2019 LA

```bash
python utils/tools/cul_eer.py --scoreFile <path>/infer_19.log
```

### ASVspoof2021 LA

```bash
python utils/tools/cul_eer21.py --scoreFile <path>/infer_LA21.log
```

### ASVspoof2021 DF

```bash
python utils/tools/cul_eer21df.py --scoreFile <path>/infer_DF21.log
```

### In-the-Wild (ITW)

```bash
python utils/tools/cul_itw.py --scoreFile <path>/infer_ITW.log
```

These scripts compute:

* Equal Error Rate (EER)
* Dataset-specific performance metrics
* Uncertainty-related evaluation metrics

---

## 4. MoE Workflow Summary

```text
Dataset Path Setup
        ↓
Training (DA5)
        ↓
Saved MoE Checkpoints
        ↓
Inference / Testing (DA3 or DA7)
        ↓
Score Files (infer_*.log)
        ↓
Evaluation & Uncertainty Metrics
```

This workflow is shared across all supported datasets and enables reproducible MoE-based anti-spoofing experiments.

---

## 5. Pretrained Model Download (Required)

The pretrained model used in the MoE experiments (**Wav2Vec2-XLS-R-300M**) is **too large to be included in this repository**.

Therefore, users must **manually download** the pretrained model files before running training or inference.

### Download Instructions

Please download all files from the following Hugging Face repository:

```
https://huggingface.co/facebook/wav2vec2-xls-r-300m/tree/main
```

### Required Directory Structure

After downloading, place the files under the following directory:

```text
moe/data/pretrained_model/facebook/wav2vec2-xls-r-300m/
```

The final directory should contain the same files as provided in the Hugging Face link (e.g., model weights, configuration files, tokenizer files, etc.).

⚠️ **Important**: Do not rename the folder. The code assumes the pretrained model is located exactly at the path shown above.

Once this directory is correctly set up, the MoE training and inference scripts will automatically load the pretrained model without further configuration.
