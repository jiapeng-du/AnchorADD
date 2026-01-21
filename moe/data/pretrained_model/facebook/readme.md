# Pretrained Model Setup

This project requires a large pretrained model (**Wav2Vec2-XLS-R-300M**) for MoE-based experiments. Due to its size, the pretrained model is **not included** in this repository and must be downloaded manually.

---

## 1. Required Pretrained Model

* **Model name**: Wav2Vec2-XLS-R-300M
* **Provider**: Facebook AI / Meta
* **Hosting platform**: Hugging Face

Official download link:

```text
https://huggingface.co/facebook/wav2vec2-xls-r-300m/tree/main
```

---

## 2. Download Instructions

Please download **all files** from the Hugging Face link above. This includes (but is not limited to):

* Model weight files
* Configuration files
* Tokenizer and feature extractor files

Make sure no files are missing.

---

## 3. Directory Structure (Mandatory)

After downloading, place the pretrained model files under the following directory:

```text
moe/data/pretrained_model/facebook/wav2vec2-xls-r-300m/
```

The directory contents should exactly match the files provided in the Hugging Face repository.

⚠️ **Important Notes**:

* Do **NOT** rename the directory
* Do **NOT** change the directory depth
* The code assumes this exact path when loading the pretrained model

If the directory structure is incorrect, training and inference will fail.

---

## 4. Usage

Once the pretrained model is correctly placed in the directory above:

* No additional configuration is required
* The MoE training and inference scripts will automatically load the pretrained model

You can now proceed to follow the MoE experiment README to run training and testing.

---

## 5. Common Issues

* **File not found / HFValidationError**:
  Ensure all files from the Hugging Face repository are downloaded and placed in the correct directory.

* **Wrong path error**:
  Double-check the directory name and path spelling.

---

This README is intentionally separated from the main experiment instructions to keep model setup clear and explicit.
