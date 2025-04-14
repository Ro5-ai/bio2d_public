open('README.md','w', encoding='utf-8').write('# Benchmarking ML in ADMET Predictions

This repository contains the code accompanying the paper:

**"Benchmarking ML in ADMET Predictions: The Practical Impact of Feature Representations in Ligand-Based Models"**

## Abstract

This study, focusing on predicting Absorption, Distribution, Metabolism, Excretion, and Toxicology (ADMET) properties, addresses the key challenges of ML models trained using ligand-based representations. We propose a structured approach to data feature selection, taking a step beyond the conventional practice of combining different representations without systematic reasoning. Additionally, we enhance model evaluation methods by integrating cross-validation with statistical hypothesis testing, adding a layer of reliability to the model assessments. Our final evaluations include a practical scenario, where models trained on one source of data are evaluated on a different one. This approach aims to bolster the reliability of ADMET predictions, providing more dependable and informative model evaluations.

## Introduction

This codebase allows you to train and evaluate machine learning models for predicting ADMET properties using various molecular feature representations. By following the instructions below, you can reproduce the results from the paper; in particular, the test results of various model configurations in Table 9, as well as any model + set of features on any dataset.

## Installation

### Prerequisites

- Conda (or Miniconda) installed on your system.

### Steps

1. **Clone the repository**

   ```bash
git clone https://github.com/2dbio_public.git
cd bio2d_public
   ```

2. **Download data**

   ```bash
wget https://ro5-public.s3.amazonaws.com/admet_datasets.zip
unzip admet_datasets.zip
   ```

3. **Set up the Virtual Environment**

   ```bash
conda create -n b2d python=3.9
conda activate b2d
pip install -e .
   ```

## Usage

This repository contains several scripts that allow you to train and evaluate models in different scenarios. Below are instructions for the two main scripts.

---

### Script: `bin/train_and_evaluate_model.py`

This script trains and evaluates a machine learning model for an ADMET dataset based on your chosen parameters.

#### Command-Line Arguments

- **`--dataset`**: The dataset to be used for training and evaluation.  
  *Choices:* `bioavailability_ma`, `hia_hou`, `pgp_broccatelli`, `bbb_martins`, `cyp2c9_veith`, `cyp2d6_veith`, `cyp3a4_veith`, `cyp2c9_substrate_carbonmangels`, `cyp2d6_substrate_carbonmangels`, `cyp3a4_substrate_carbonmangels`, `herg`, `ames`, `dili`, `caco2_wang`, `lipophilicity`, `ppbr_az`, `ld50_zhu`, `vdss_lombardo`, `half_life_obach`, `clearance_microsome_az`, `nih_solubility`, `rlm`, `solubility`, `hlm`, `mdr1-mdck`

- **`--model_type`**: Type of machine learning model to use.  
  *Choices:* `random_forest`, `lightgbm`, `support_vector_machine`, `catboost`, `mpnn`

- **`--feature_type`**: Type of molecular feature representation.  
  *Choices:* `atom_pair`, `ecfp4`, `rdkit_desc`, `mordred`, `maccs_keys`, `erg`, `avalon`, `mol2vec`, `megamolbart`, `bartsmiles`, `grover`, `molformer`

- **`--optimized_hyperparameters`**: Whether to use optimized hyperparameters (specifically for CatBoost).  
  *Choices:* `True`, `False`

- **`--use_precomputed`**: Whether to use pre-computed features or compute them anew.  
  Note: Only standard cheminformatics features can be computed on the fly; deep learning-based representations (e.g. `megamolbart`, `molformer`, `bartsmiles`, `grover`) must be precomputed.  
  *Choices:* `True`, `False`

#### Basic Usage Example

```bash
python bin/train_and_evaluate_model.py \ 
    --dataset DATASET_NAME \ 
    --model_type MODEL_TYPE \ 
    --feature_type FEATURE_TYPE \ 
    --optimized_hyperparameters True
```

For example, to run CatBoost with rdkit_desc features using unoptimized hyperparameters:

```bash
python bin/train_and_evaluate_model.py \ 
    --dataset bbb_martins \ 
    --model_type catboost \ 
    --feature_type rdkit_desc.ecfp4 \ 
    --optimized_hyperparameters False
```

---

### Script: `domain_adaptation.py`

This script is designed to perform domain adaptation experiments by training a model on source data and adapting it to a different domain. In addition to standard adaptation methods (e.g., `TrAdaBoostR2`, `KMM`, `KLIEP`), two additional adapter methods—**CombinedDataAdapter** and **ContinuedTrainingAdapter**—are provided.

#### Command-Line Arguments

- **`--source_data_path`**: Path to the source dataset pickle file.  
  *(Default: `/home/ubuntu/bio2d_public/data/clean_nih_sol_fp.pkl`)*

- **`--domain_data_path`**: Path to the domain dataset pickle file.  
  *(Default: `/home/ubuntu/bio2d_public/data/clean_biogen_sol_fp.pkl`)*

- **`--output_path`**: Full path for the output CSV file. If not provided, the script automatically generates a default name based on key input parameters.

- **`--output_folder`**: Folder in which to save the output CSV when `--output_path` is not specified.  
  *(Default: `./output`)*

- **`--dataset_name`**: Dataset name used to load the configuration (e.g., `solubility`, `hppb`, `hlm`).

- **`--adaptation_method`**: The adaptation method to use.  
  *Options:* `TrAdaBoostR2`, `TwoStageTrAdaBoostR2`, `NearestNeighborsWeighting`, `KMM`, `KLIEP`, `CombinedDataAdapter`, `ContinuedTrainingAdapter`

- **`--num_fractions`**: Number of fractions into which the domain training set is divided.  
  *(Default: `10`)*

- **`--num_scaffold_splits`**: Number of scaffold splits to generate from the domain dataset.  
  *(Default: `5`)*

- **`--num_processes`**: Number of processes to use for parallel training.  
  *(Default: `4`)*

- **Adapter-Specific Hyperparameters:**
  - **`--adapter_n_estimators`**: Number of estimators for the adaptation method.  
    *(Default: `10`)*
  - **`--adapter_random_state`**: Random seed for the adaptation method.  
    *(Default: `0`)*
  - **`--additional_iterations`**: Additional iterations for continued training (used only by ContinuedTrainingAdapter).  
    *(Default: `1000`)*

#### Default Output File Naming

If you do not supply an explicit `--output_path`, the script automatically generates a default file name using the following format:

```
{dataset_name}_{adaptation_method}_splits{num_scaffold_splits}_iters{iterations}.csv
```

For example, if you run with:
- `--dataset_name hppb`
- `--adaptation_method ContinuedTrainingAdapter`
- `--num_scaffold_splits 5`
- And the CatBoost iteration count (from the dataset configuration) is `2000`,

then the default output file name will be:

```
hppb_ContinuedTrainingAdapter_splits5_iters2000.csv
```

and the file will be placed in the folder specified by `--output_folder` (defaulting to `./output`).

#### Basic Usage Example

```bash
python domain_adaptation.py \ 
    --dataset_name hppb \ 
    --adaptation_method ContinuedTrainingAdapter \ 
    --num_scaffold_splits 5 \ 
    --num_fractions 5 \ 
    --additional_iterations 1000
```

This command runs the domain adaptation experiment on the `hppb` dataset using the `ContinuedTrainingAdapter` method with 5 scaffold splits and 5 fractions.

---

## Notes

- **Datasets & Resources:**  
  Ensure that all required datasets and resources are available in the appropriate directories.

- **Performance Metrics:**  
  The training outputs include key performance metrics.
  - *For Binary Classification:* accuracy, F1 score, recall, precision, ROC-AUC, and AU-PRC.
  - *For Regression:* Pearson and Spearman correlations, RMSE, MAE, and normalized RMSE.

- **Reproducibility:**  
  Random states and seeds are configurable to help reproduce your results across different runs.

