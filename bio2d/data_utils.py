import deepchem as dc
import pandas as pd
from rdkit import Chem


def smiles_to_deepchem_dataset(df, smiles_col):
    """Convert a DataFrame with SMILES strings to a DeepChem Dataset."""
    mols = [Chem.MolFromSmiles(smile) for smile in df[smiles_col]]

    # Featurize the molecules
    featurizer = dc.feat.ConvMolFeaturizer()
    features = featurizer.featurize(mols)

    # Create a DeepChem dataset with SMILES as IDs
    dataset = dc.data.NumpyDataset(X=features, ids=df[smiles_col].values)

    return dataset


def split_dataframe(df, smiles_col, train_dataset, val_dataset):
    """Split the original DataFrame using the datasets."""
    # The ids in the datasets are the SMILES strings, so we can use them to filter the original dataframe
    train_df = df[df[smiles_col].isin(train_dataset.ids)]
    val_df = df[df[smiles_col].isin(val_dataset.ids)]
    return train_df, val_df


def split_by_scaffold(df, smiles_col):
    """Split the dataset by scaffold."""
    dataset = smiles_to_deepchem_dataset(df, smiles_col)
    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, val_dataset = splitter.train_test_split(dataset, frac_train=0.75)

    # Split the original DataFrame
    train_df, val_df = split_dataframe(df, smiles_col, train_dataset, val_dataset)
    return train_df, val_df


def split_by_fingerprint(df, smiles_col):
    """Split the dataset by fingerprint."""
    dataset = smiles_to_deepchem_dataset(df, smiles_col)
    splitter = dc.splits.FingerprintSplitter()
    train_dataset, val_dataset = splitter.train_test_split(dataset, frac_train=0.75)

    # Split the original DataFrame
    train_df, val_df = split_dataframe(df, smiles_col, train_dataset, val_dataset)
    return train_df, val_df


def split_randomly(df, smiles_col):
    """Split the dataset randomly."""
    dataset = smiles_to_deepchem_dataset(df, smiles_col)
    splitter = dc.splits.RandomSplitter()
    train_dataset, val_dataset = splitter.train_test_split(dataset, frac_train=0.75)

    # Split the original DataFrame
    train_df, val_df = split_dataframe(df, smiles_col, train_dataset, val_dataset)
    return train_df, val_df
