# Autoencoder vs Variational Autoencoder for Fraud Detection

Comparing a standalone Autoencoder (AE) and a beta Variational Autoencoder (beta-VAE) for unsupervised anomaly detection on financial transactions, using the IEEE-CIS and Bank Account Fraud (BAF) datasets.

This repository holds the code behind every table and figure in the study. The models, the training settings and the evaluation logic are exactly the ones that produced the reported results. Compared with the working notebooks, only headers, comments and the Colab folder path were tidied.

## What the code does

1. Prepares two public datasets, IEEE-CIS Fraud Detection and the Bank Account Fraud (BAF) Base variant, with a time-based split. The models are trained on normal records only.
2. Trains an Autoencoder (AE) and a beta Variational Autoencoder (beta-VAE) under identical conditions, with five seeds each.
3. Scores every record by its reconstruction error and compares the two models with each other and with Principal Component Analysis (PCA) and Isolation Forest.
4. Compares fixed and calibrated thresholds, runs the statistical tests and the sensitivity check, and looks at where the reconstruction error comes from.

## Run order

| Step | Notebook | Run on | What it produces |
|---|---|---|---|
| 1 | `01_Data_Preparation.ipynb` | Laptop | Preprocessed arrays in `Processed/`, EDA figures (Figures 4.1 to 4.7) |
| 2 | `02_AE_VAE_Training.ipynb` | Colab GPU (laptop works, slower) | 28 training runs in `Results/runs_final/`, latent size search, training curves (Figure 4.9) |
| 3 | `02b_Sensitivity_Check.ipynb` | Colab GPU | 18 sensitivity runs in `Results/runs_sensitivity/` and the sensitivity table |
| 4 | `03_Evaluation.ipynb` | Laptop | All main results tables, ROC/PR curves, confusion matrices, error distributions (Figures 5.1, 5.4, 5.5) |
| 5 | `04_Error_Analysis.ipynb` | Laptop | Error analysis tables and Figure 5.8 |
| 6 | `05_Thesis_Figures.ipynb` | Laptop | Figures 4.8, 5.2, 5.3, 5.6, 5.7, 5.9 and 5.10, built only from saved results |

`thesis_models.py` holds the shared model code (AE, VAE, loss, anomaly score, training loop). Notebooks 02, 02b and 04 import it, so every notebook uses exactly the same models. 
Notebooks 03, 04 and 05 do not train anything. They read the saved scores and model weights, so they can be re-run in a few minutes to check any number in Chapter 5.

## Folder layout

The notebooks expect this layout, with paths taken relative to the notebook folder (`..`):

```
04_Final_Thesis/
  Dataset/        raw zip files (downloaded from Kaggle, not included)
  Processed/      written by Notebook 01
  Results/        written by Notebooks 02, 02b, 03 and 04
  Figures/        written by Notebooks 01 to 05 (created automatically)
  Final_Code/     this folder
```

## Data

The datasets are not included because of their licences. Download them from Kaggle and place the zip files in `Dataset/`:

* IEEE-CIS Fraud Detection: https://www.kaggle.com/competitions/ieee-fraud-detection/data
  Download `ieee-fraud-detection.zip`; only `train_transaction.csv` and `train_identity.csv` are used. You need to sign in to Kaggle and accept the competition rules before the download button appears.
* Bank Account Fraud Dataset Suite (NeurIPS 2022): https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
  Save the download as `dataset__NeurIIPS.zip`; only `Base.csv` is used.

## Environment

Python 3 with numpy, pandas, scikit-learn, scipy, matplotlib, seaborn and PyTorch. The code uses the GPU when one is available and the CPU otherwise (`thesis_models.DEVICE`), so the same notebooks run on a laptop and on Google Colab.

## Running on Google Colab

1. Upload the `04_Final_Thesis` folder (with `Processed/`, `Results/` and `Final_Code/`) to `My Drive`.
2. Open Notebook 02 or 02b in Colab and choose Runtime, Change runtime type, T4 GPU.
3. Run all cells. The notebook mounts Google Drive and finds the folder by itself. Finished runs are skipped, so it is safe to restart.

## Reproducibility

Every run sets its seed for Python, NumPy and PyTorch. GPU training is not perfectly deterministic, so re-training can change the last decimals of a result. The saved scores in `Results/` are the ones reported in the thesis, and Notebooks 03 to 05 reproduce every table from them exactly.
