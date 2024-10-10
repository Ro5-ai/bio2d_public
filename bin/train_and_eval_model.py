import argparse
import os
import json
from bio2d.model_training import train_and_evaluate_model
from bio2d.features.feature_configs import get_scaling_config

from bio2d.features.pipeline import FeaturizationPipeline
from pathlib import Path

from sklearn.preprocessing import RobustScaler

import pandas as pd
import numpy as np


DATASET_TO_TASK_TYPE = {
    'bioavailability_ma': 'binary',
    'hia_hou': 'binary',
    'pgp_broccatelli': 'binary',
    'bbb_martins': 'binary',
    'cyp2c9_veith': 'binary',
    'cyp2d6_veith': 'binary',
    'cyp3a4_veith': 'binary',
    'cyp2c9_substrate_carbonmangels': 'binary',
    'cyp2d6_substrate_carbonmangels': 'binary',
    'cyp3a4_substrate_carbonmangels': 'binary',
    'herg': 'binary',
    'ames': 'binary',
    'dili': 'binary',
    'caco2_wang': 'regression',
    'lipophilicity': 'regression',
    'ppbr_az': 'regression',
    'ld50_zhu': 'regression',
    'vdss_lombardo': 'regression',
    'half_life_obach': 'regression',
    'clearance_microsome_az': 'regression',
    'nih_solubility': 'regression',
    'rlm': 'regression',
    'solubility': 'regression',
    'hlm': 'regression',
    'mdr1-mdck': 'regression'
}

FINGERPRINT_FEATURES = ['ecfp4', 'atom_pair', 'rdkit_fp', 'avalon']

LOG_TRANSFORM_DATASETS = ['half_life_obach', 'vdss_lombardo', 'clearance_hepatocyte_az', 'clearance_microsome_az']

OPTIMAL_FEATURES = set(['rdkit_desc', 'ecfp4', 'erg', 'avalon'])

DEEP_LEARNING_FEATURES = ['molformer', 'bartsmiles', 'megamolbart', 'grover']


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train and evaluate a model on a benchmark",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hia_hou",
        help="Dataset to use for training"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="mpnn",
        help="Type of model to train (e.g., catboost, random_forest, etc.)"
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="molformer",
        help="Type of features to use (e.g., rdkit_desc, ecfp4, etc.)"
    )
    parser.add_argument(
        "--optimized_hyperparameters",
        type=bool,
        default="False",
        help="Whether to use the optimized hyperparameters (only applicable to CatBoost models with the optimized features)"
    )
    parser.add_argument(
        "--use_precomputed",
        type=bool,
        default="True",
        help="Whether to use pre-computed features or compute them anew. Note this is only implemented for standard cheminformatics features, the deep learning ones need to be loaded"
    )
    return parser.parse_args()


def prepare_data(features, features_to_drop, dataset):
    # Check and drop only the columns that are present in the DataFrame
    features_to_drop = [col for col in features_to_drop if col in features.columns]
    X = features.drop(columns=features_to_drop)

    if dataset in LOG_TRANSFORM_DATASETS:
        y = features['value'].apply(np.log10).to_numpy()
    else:
        y = features['value'].to_numpy()
    return X, y


def process_features(train_features, val_features, dataset, scaler=None):
    cols_to_drop = ['smiles', 'value']

    X_train, y_train = prepare_data(train_features, cols_to_drop, dataset)
    X_val, y_val = prepare_data(val_features, cols_to_drop, dataset)

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    train_data = {'smiles': None, 'features': X_train, 'values': y_train}
    test_data = {'smiles': None, 'features': X_val, 'values': y_val}

    return train_data, test_data


def check_inputs(feature_type, use_precomputed):
    if use_precomputed:
        return True
    ftl = feature_type.split('.')
    for ft in ftl:
        if ft in DEEP_LEARNING_FEATURES:
            raise AssertionError(f'Deep learning features ({ft}) need to be loaded, as the models are not installed in this repository.')


def load_precomputed_features(base_path, feature_type, dataset):
    all_train_data = []
    all_test_data = []
    for ff in feature_type.split('.'):
        train_features = pd.read_csv(os.path.join(base_path, 'data', dataset, 'features', 'train', f'{feature_type}.csv'))
        val_features = pd.read_csv(os.path.join(base_path, 'data', dataset, 'features', 'val', f'{feature_type}.csv'))
        test_features = pd.read_csv(os.path.join(base_path, 'data', dataset, 'features', 'test', f'{feature_type}.csv'))
        train_val_features = pd.concat((train_features, val_features), axis=0)

    if args.feature_type not in FINGERPRINT_FEATURES:
        scaler = RobustScaler()
        train_data, test_data = process_features(train_val_features, test_features, dataset, scaler)
    else:
        train_data, test_data = process_features(train_val_features, test_features, dataset)

    all_train_data.append(train_data)
    all_test_data.append(test_data)

    final_train_data = {
        'smiles': list(train_df['smiles']),
        'features': np.hstack([ff['features'] for ff in all_train_data]),
        'values': all_train_data[0]['values'],
    }

    final_test_data = {'smiles': list(test_df['smiles']), 'features': np.hstack([ff['features'] for ff in all_test_data]), 'values': all_test_data[0]['values']}
    return final_train_data, final_test_data


def compute_features(feature_type):
    fmm = get_scaling_config(feature_type.split('.'))
    fp = FeaturizationPipeline(fmm)

    X_train, _ = fp.fit_transform(X_train, y_train)
    X_test, _ = fp.transform(test_df['smiles'])

    final_train_data = {'smiles': train_df['smiles'], 'features': X_train, 'values': y_train}
    final_test_data = {'smiles': test_df['smiles'], 'features': X_test, 'values': test_df['value']}

    return final_train_data, final_test_data


if __name__ == "__main__":
    args = get_args()

    model_type = args.model_type
    feature_type = args.feature_type
    dataset = args.dataset
    task_type = DATASET_TO_TASK_TYPE[dataset]
    use_precomputed = args.use_precomputed

    check_inputs(feature_type, use_precomputed)

    base_path = Path(__file__).resolve().parent.parent

    train_data_path = os.path.join(base_path, 'data', dataset, 'data', 'train_val.csv')
    test_data_path = os.path.join(base_path, 'data', dataset, 'data', 'test.csv')
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    X_train = train_df['smiles']
    y_train = train_df['value']

    if not use_precomputed:
        final_train_data, final_test_data = compute_features(feature_type)
    else:
        final_train_data, final_test_data = load_precomputed_features(base_path, feature_type, dataset)


    # This is a set of hyperparameters found using a standard grid search, for each dataset separately, 
    # and only for the specific optimal combination of features `OPTIMAL_FEATURES`
    hp = None
    if args.optimized_hyperparameters and model_type == 'catboost' and set(feature_type.split('.')) == OPTIMAL_FEATURES:
        config_path = base_path / "configs" / "catboost_hyperparameters.json"
        with open(config_path, 'r') as f:
            hp = json.load(f)[dataset]

    train_and_evaluate_model(
        task_type=task_type,
        model_type=model_type,
        train_data=final_train_data,
        test_data=final_test_data,
        hyperparameters=hp
    )
