# Dataset Preparation

This repository uses several public speech deepfake datasets.
Due to their large size, **datasets are NOT included** in this repository and must be **downloaded manually**.

After downloading, please place all datasets under the `data/` directory following the structure described below.

---

## Supported Datasets and Download Links

### ASVspoof 2019 Logical Access (LA)

* **Directory name**: `ASVspoof2019_LA`
* **Download link**:
  [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)

---

### ASVspoof 2021 Logical Access (LA) – Evaluation Set

* **Directory name**: `ASVspoof2021_LA_eval`
* **Download link**:
  [https://zenodo.org/records/4837263](https://zenodo.org/records/4837263)

---

### ASVspoof 2021 DeepFake (DF) – Evaluation Set

* **Directory name**: `ASVspoof2021_DF_eval`
* **Download link**:
  [https://zenodo.org/records/4835108](https://zenodo.org/records/4835108)

---

### In-The-Wild (ITW) Dataset

* **Directory name**: `release_in_the_wild`
* **Download link**:
  [https://deepfake-total.com/in_the_wild](https://deepfake-total.com/in_the_wild)

---

## ASVspoof 2021 Keys (Labels) and Metadata

For ASVspoof 2021 datasets (LA and DF), the **official keys (labels) and metadata** are provided by the ASVspoof organizers:

* [https://www.asvspoof.org/index2021.html](https://www.asvspoof.org/index2021.html)

These files are required for evaluation and protocol preparation.

---

## Required Directory Structure

After downloading and extracting all datasets, the `data/` directory should be organized as follows:

```
data/
├── ASVspoof2019_LA/
├── ASVspoof2021_LA_eval/
├── ASVspoof2021_DF_eval/
└── release_in_the_wild/
```

---

## Dataset Splitting and Path Configuration

This repository provides a script to split datasets by **splitting protocol files** and **updating audio paths**:

```bash
python split_datasets.py
```

### Functionality of `split_datasets.py`

* Splits protocol files into training / development / evaluation subsets (depending on the dataset)
* Updates absolute or relative audio paths in protocol files
* Ensures protocol consistency across experiments

⚠️ **Important**:
Before running experiments, please ensure that:

* All dataset paths are correctly set
* Audio files and protocol files are properly aligned

---

## Notes

* All datasets are released for research purposes only.
* Users are responsible for downloading datasets and organizing them correctly.
* Experiments may fail if dataset paths or protocol files are incorrectly configured.

