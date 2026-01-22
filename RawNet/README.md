# RawNet Experiment Usage

This directory contains the RawNet-based baseline used in this repository.
Different entry scripts are used depending on the dataset, following the official ASVspoof RawNet2 baseline design.

The implementation and usage are adapted from the official ASVspoof 2021 RawNet2 baseline:

* [https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)

---

## Entry Scripts

Two different entry scripts are used:

* `main.py`
  Used **only for ASVspoof 2019 LA** experiments.

* `main2.py`
  Used for **all other datasets**, including:

  * ASVspoof 2021 LA
  * ASVspoof 2021 DF
  * In-The-Wild (ITW)

---

## Training

### ASVspoof 2019 LA

To train RawNet on ASVspoof 2019 LA, use `main.py`:

```bash
python main.py [other arguments]
```

No evaluation flags are required during training.

---

### ASVspoof 2021 LA / DF and ITW

To train RawNet on ASVspoof 2021 LA, ASVspoof 2021 DF, or ITW, use `main2.py`:

```bash
python main2.py [other arguments]
```

No evaluation-specific flags are required during training.

---

## Evaluation / Testing

For **evaluation or testing**, additional flags must be enabled.

### Required Evaluation Flags

When running evaluation, **all datasets** must include the following arguments:

* `--is_eval`
* `--eval`
* `--model_path`

These flags indicate evaluation mode and specify the trained model checkpoint.

---

### Example: Evaluation Command

```bash
python main2.py \
  --is_eval \
  --eval \
  --model_path path/to/checkpoint.pth \
  [other arguments]
```

> Note: The same evaluation flags apply when using `main.py` for ASVspoof 2019 LA evaluation.

---

## Uncertainty Quantification (UQ) and Data Augmentation (DA)

Uncertainty Quantification (UQ) and Data Augmentation (DA) strategies are **controlled inside the codebase**.

* UQ-related behavior is enabled or disabled by modifying the corresponding UQ flags or modules in the code
* DA strategies (e.g., RawBoost variants) are configured through code-level switches or configuration arguments

Please refer to the source code for the exact implementation details and available options.

---

## Notes

* Training **does not require** evaluation flags.
* Evaluation **must include** `--is_eval`, `--eval`, and `--model_path`.
* Always ensure dataset paths and protocol files are correctly configured before running experiments.

This design follows the official ASVspoof RawNet2 baseline structure for reproducibility and consistency.
