from bio2d.features.featurize import features_from_smiles
from bio2d.features.engineering import BaseFeatureTransformer
from bio2d.features.feature_configs import get_scaling_config, FeatureManipulationMethods
from typing import Dict, Tuple
import pandas as pd
from copy import deepcopy
import os
from bio2d.features.featurize import features_from_smiles


class FeaturizationPipeline:
    def __init__(self, feature_manipulation_config):
        self.methods = feature_manipulation_config
        self.eval = eval
        self.fit_method_dict = {}

    def fit(self, smiles_list, y=None):
        X = features_from_smiles(smiles_list, self.methods.feature_list)
        for name, method_class in self.methods.__dict__.items():
            if type(method_class) is not list and method_class is not None:
                assert issubclass(method_class, BaseFeatureTransformer)
                method_instance = method_class()
                method_instance.fit(X, y)
                self.fit_method_dict[name] = method_instance
        return X

    def transform(self, smiles_list, y=None, eval=True):
        X = features_from_smiles(smiles_list, self.methods.feature_list)

        fmd = deepcopy(self.fit_method_dict)
        if eval:
            fmd['upsampling'] = None

        for _, method_instance in fmd.items():
            if method_instance is not None:
                X, y = method_instance.transform(X, y)
        return X, y

    def transform_from_features(self, X, y=None):
        for _, method_instance in self.fit_method_dict.items():
            if method_instance is not None:
                X, y = method_instance.transform(X, y)
        return X, y

    def fit_transform(self, X, y):
        feat_df = self.fit(X, y)
        return self.transform_from_features(feat_df, y)


def get_scaled_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    s3_uri: str,
    feature_type: str,
    feature_manipulation_config: FeatureManipulationMethods = None,
    combine_data: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Compute and store features for the training and test datasets of a benchmark.

    Args:
        train_df (pd.DataFrame): Training data containing 'smiles' and 'value'.
        test_df (pd.DataFrame): Test data containing 'smiles' and 'value'.
        s3_uri (str): The URI for storing data in S3.
        feature_type (str): The type of features to compute.
        feature_manipulation_config (Union[Dict, None], optional): Configuration for feature manipulation. Defaults to None.

    Returns:
        Tuple[Dict, Dict]: Training and test data with features and values.
    """

    smiles_list = train_df['smiles'].to_list()
    train_values = train_df['value'].to_list()

    if feature_manipulation_config is None:
        feature_list = feature_type.split('.')
        feature_manipulation_config = get_scaling_config(feature_list)

    fp = FeaturizationPipeline(feature_manipulation_config=feature_manipulation_config)

    file_suffix = "combined_data_" if combine_data else ""
    fp_file_name = f"{file_suffix}scaling_feature_pipeline_{feature_type}.pkl"
    feature_pipeline_path = os.path.join(s3_uri, fp_file_name)

    if not combine_data:
        train_features, train_values = fp.fit_transform(smiles_list, train_values)
        train_data = {'features': train_features.to_numpy(), 'values': train_values}

        object_to_s3(fp, s3_uri, fp_file_name)

        test_features, _ = fp.transform(test_df["smiles"].to_list(), eval=True)
        test_data = {'features': test_features.to_numpy(), 'values': test_df['value'].to_list()}

        return train_data, test_data, feature_pipeline_path

    full_data = pd.concat([train_df, test_df], ignore_index=True)
    smiles_list = full_data['smiles'].to_list()
    full_values = full_data['value'].to_list()
    full_features, _ = fp.fit_transform(smiles_list, train_values)
    full_data = {'features': full_features.to_numpy(), 'values': full_values}

    object_to_s3(fp, s3_uri, fp_file_name)

    return full_data, None, feature_pipeline_path


def get_concatenated_features(train_df: pd.DataFrame, test_df: pd.DataFrame, s3_uri: str, feature_type: str, combine_data: bool = True):
    """
    Compute and concatenate a list of features for the training and test datasets of a benchmark.

    Args:
        train_df (pd.DataFrame): Training data containing 'smiles' and 'value'.
        test_df (pd.DataFrame): Test data containing 'smiles' and 'value'.
        s3_uri (str): The URI for storing data in S3.
        feature_type (str): The type of features to concatenate.

    Returns:
        Tuple[Dict, Dict]: Training and test data with concatenated features and values.
    """

    feature_list = feature_type.split('.')

    train_feat = features_from_smiles(train_df['smiles'], feature_list, s3_uri, test=False)
    test_feat = features_from_smiles(test_df['smiles'], feature_list, s3_uri, test=True)

    train_data = {'features': train_feat, 'values': train_df['value'].to_list()}
    test_data = {'features': test_feat, 'values': test_df['value'].to_list()}

    return train_data, test_data


def compute_and_store_features(entry_list, feature_type_list, split_type):
    """
    Computes and stores features for given entries and feature types,
    organized by specified split type.

    Args:
        entry_list (list): List of entries to process. Each entry should have
                           attributes 'uuid', 's3_uri', and 'dataset'.
        feature_type_list (list): List of feature types to compute for each entry.
        split_type (str): The type of split (e.g., 'train_val_test') to consider
                          when organizing and saving features.
    """

    for entry in entry_list:
        print('------------------------------------------------------------------')
        print(f'Computing & storing features for {entry.dataset}...')

        splits_folder = os.path.join(entry.s3_uri, 'train_val_splits', split_type)
        features_folder = os.path.join(entry.s3_uri, 'features', split_type)

        # Load dataset splits
        train = pd.read_csv(os.path.join(splits_folder, 'train.csv'))
        val = pd.read_csv(os.path.join(splits_folder, 'val.csv'))
        test = pd.read_csv(os.path.join(entry.s3_uri, 'data', 'test_clean.csv'))

        for feature_type in feature_type_list:
            print(feature_type)

            # Compute features for each split
            for split_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
                features = features_from_smiles(dataset['smiles'].to_list(), [feature_type])
                features['value'] = dataset['value']
                features.to_csv(os.path.join(features_folder, split_name, f'{feature_type}.csv'), index=False)
