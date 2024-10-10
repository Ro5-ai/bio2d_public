
# Benchmarking ML in ADMET Predictions

This repository contains the code accompanying the paper:

**"Benchmarking ML in ADMET Predictions: The Practical Impact of Feature Representations in Ligand-Based Models"**

## Abstract

This study, focusing on predicting Absorption, Distribution, Metabolism, Excretion, and Toxicology (ADMET) properties, addresses the key challenges of ML models trained using ligand-based representations. We propose a structured approach to data feature selection, taking a step beyond the conventional practice of combining different representations without systematic reasoning. Additionally, we enhance model evaluation methods by integrating cross-validation with statistical hypothesis testing, adding a layer of reliability to the model assessments. Our final evaluations include a practical scenario, where models trained on one source of data are evaluated on a different one. This approach aims to bolster the reliability of ADMET predictions, providing more dependable and informative model evaluations.

## Introduction

This codebase allows you to train and evaluate machine learning models for predicting ADMET properties using various molecular feature representations. By following the instructions below, you can reproduce the results from the paper; in particular, the test results of various model configurations in Table 9, as well as any model + set of features on any dataset.

## Installation

### Prerequisites

- Python 3.9 installed on your system.

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/2dbio_public.git
   cd 2dbio_public
   ```

2. **Download data**

   ```bash
   wget https://ro5-public.s3.amazonaws.com/admet_datasets.zip
   unzip admet_datasets.zip
   ```

2. **Virtual environment**

   ```bash
   conda create -n b2d python=3.9
   conda activate b2d
   pip install -e .
   ```


## Usage

### Script: `bin/train_and_evaluate_model.py`

This script trains and evaluates a machine learning model for an ADMET dataset based on your chosen parameters.

### Command-Line Arguments

- `--dataset`: The dataset to be used for training and evaluation.
  - Choices: `bioavailability_ma`, `hia_hou`, `pgp_broccatelli`, `bbb_martins`, 
    `cyp2c9_veith`, `cyp2d6_veith`, `cyp3a4_veith`, `cyp2c9_substrate_carbonmangels`, 
    `cyp2d6_substrate_carbonmangels`, `cyp3a4_substrate_carbonmangels`, `herg`, 
    `ames`, `dili`, `caco2_wang`, `lipophilicity`, `ppbr_az`, `ld50_zhu`, 
    `vdss_lombardo`, `half_life_obach`, `clearance_microsome_az`, `nih_solubility`, 
    `rlm`, `solubility`, `hlm`, `mdr1-mdck`
- `--model_type`: Type of machine learning model to use.
  - Choices: `random_forest`, `lightgbm`, `support_vector_machine`, `catboost`, `mpnn`
- `--feature_type`: Type of molecular feature representation.
  - Choices: `atom_pair`, `ecfp2`, `rdkit_desc`, `mordred`, `maccs_keys`, `erg`, `avalon`, `mol2vec`, `megamolbart`, `bartsmiles`, `grover`, `molformer`
- `--optimized_hyperparameters`: Whether to use optimized hyperparameters (specifically for `CatBoost`).
  - Choices: `True`, `False`
- `--use_precomputed`: Whether to use pre-computed features or compute them anew. Note this is only implemented for standard cheminformatics features, the deep learning ones need to be loaded (i.e. use_precomputed = True).
  - Choices: `True`, `False`


### Basic Usage

```bash
python bin/train_and_evaluate_model.py \
    --dataset DATASET_NAME
    --model_type MODEL_TYPE \
    --feature_type FEATURE_TYPE \
    --optimized_hyperparameters True|False \

```

### Examples

- **CatBoost with rdkit_desc features with unoptimized hyperparameters**

  ```bash
  python bin/train_and_evaluate_model.py \
      --dataset bbb_martins \
      --model_type catboost \
      --feature_type rdkit_desc.ecfp4 \
      --optimized_hyperparameters False
  ```

- **CatBoost with rdkit_desc + ecfp4 + avalon + erg features with optimized hyperparameters**

  ```bash
  python bin/train_and_evaluate_model.py \
      --dataset ames \
      --model_type catboost \
      --feature_type rdkit_desc.ecfp4.avalon.erg \
      --optimized_hyperparameters True
  ```

- **MPNN with megamolbart features**

  ```bash
  python bin/train_and_evaluate_model.py \
      --dataset lipophilicity \
      --model_type mpnn \
      --feature_type megamolbart \
      --optimized_hyperparameters False
  ```
### Notes

- Ensure all required datasets and resources are available in the appropriate directories.
- The model training outputs basic performance metrics, including:
    - **For Binary Classification**: accuracy (`acc`), F1 score (`f1`), recall (`recall`), precision (`precision`), ROC-AUC (`rocauc`), and area under the precision-recall curve (`auprc`).
    - **For Regression**: Pearson correlation (`pearson`), Spearman correlation (`spearman`), root mean square qrror (`rmse`), mean absolute error (`mae`), and normalized RMSE (`nrmse`).

## License

This project is licensed under the MIT License.

## References

- Benchmarking ML in ADMET Predictions Paper
