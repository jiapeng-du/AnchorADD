# AnchorADD

**AnchorADD** is a research-oriented repository for audio deepfake and spoofing detection, focusing on uncertainty-aware and robustness-enhanced modeling. The repository provides **three representative baselines**, each implemented in a separate subdirectory with its own detailed usage instructions.

This README gives a high-level overview of the repository structure and how to get started.

---

## Repository Overview

The repository contains **three baseline methods**:

1. **LFCC-based baseline**
2. **MoE-based baseline**
3. **RawNet-based baseline**

Each baseline is implemented independently and can be run separately.

```text
AnchorADD/
├── lfcc/        # LFCC-based baseline
│   └── README.md
├── moe/         # Mixture-of-Experts (MoE) baseline
│   └── README.md
├── rawnet/      # RawNet-based baseline
│   └── README.md
├── requirements.txt
└── README.md    # (this file)
```

Each subdirectory contains a dedicated `README.md` that explains:

* Dataset preparation
* Training procedure
* Inference / testing procedure
* Evaluation scripts

---

## Baselines

### 1. LFCC Baseline

The **LFCC baseline** implements a traditional feature-based spoofing detection pipeline using LFCC features and classical back-end models.

To use this baseline:

```bash
cd lfcc
```

Then follow the instructions in:

```text
lfcc/README.md
```

---

### 2. MoE Baseline

The **MoE (Mixture-of-Experts) baseline** leverages deep neural networks and pretrained speech representations for uncertainty-aware spoofing detection.

To use this baseline:

```bash
cd moe
```

Then follow the instructions in:

```text
moe/README.md
```

> **Note**: The MoE baseline requires an external pretrained model that must be downloaded manually. Detailed instructions are provided in the MoE README.

---

### 3. RawNet Baseline

The **RawNet baseline** operates directly on raw waveforms and provides an end-to-end neural approach to spoofing detection.

To use this baseline:

```bash
cd rawnet
```

Then follow the instructions in:

```text
rawnet/README.md
```

---

## Dataset Preparation and Path Configuration

Before running **any** experiment, you must ensure that the dataset **protocol files** and **audio file paths** are correctly configured.

* All datasets are expected to be located under:

```text
AnchorADD/data/
```

* For each baseline, the corresponding data-loading scripts contain hard-coded or configurable paths to:

  * Protocol files (train / dev / eval splits)
  * Audio file root directories

These paths **must be updated to match your local filesystem** before running training or inference.

⚠️ **Important Notes**:

* Some datasets may be **too large to be included** in the repository
* Such datasets need to be **downloaded manually** and **properly organized/split** under `AnchorADD/data/`
* Experiments will fail if protocol paths or audio paths are incorrect

Please refer to the baseline-specific README files for detailed instructions on where and how to modify dataset paths.

---

## Environment Setup

All experiments must be run under an environment that is **consistent with the provided dependencies**.

Before running any baseline, please install the required packages:

```bash
pip install -r requirements.txt
```

⚠️ **Important**:

* Using a different Python version or mismatched dependencies may lead to unexpected errors
* For reproducibility, it is strongly recommended to use the exact versions listed in `requirements.txt`

---

## Getting Started

1. Clone this repository:

```bash
git clone <repository_url>
cd AnchorADD
```

2. Set up the Python environment:

```bash
pip install -r requirements.txt
```

3. Choose a baseline and follow its README:

```bash
cd lfcc   # or moe / rawnet
```

---

## Notes

* Each baseline is **self-contained** and does not interfere with others
* Dataset paths and pretrained models are handled **inside each subdirectory**
* Please refer to the corresponding README for dataset-specific and model-specific details

---

## License and Usage

This repository is intended for **research and academic use**. Please refer to the individual baseline implementations and datasets for their respective licenses.

---

If you use this repository in your research, please cite the corresponding work or baseline as described in each submodule.


