# data_utils.py

import deepchem as dc
import pandas as pd
from rdkit import Chem
import numpy as np

def smiles_to_deepchem_dataset(df, smiles_col):
    """Convert a DataFrame with SMILES strings to a DeepChem Dataset."""
    mols = [Chem.MolFromSmiles(smile) for smile in df[smiles_col]]
    featurizer = dc.feat.ConvMolFeaturizer()
    features = featurizer.featurize(mols)
    dataset = dc.data.NumpyDataset(X=features, ids=df[smiles_col].values)
    return dataset

def split_dataframe(df, smiles_col, train_dataset, val_dataset):
    """Split the original DataFrame using the dataset IDs (SMILES strings)."""
    train_df = df[df[smiles_col].isin(train_dataset.ids)]
    val_df = df[df[smiles_col].isin(val_dataset.ids)]
    return train_df, val_df

def split_by_scaffold(df, smiles_col, frac_train=0.75, seed=None):
    """Split the dataset by scaffold."""
    dataset = smiles_to_deepchem_dataset(df, smiles_col)
    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, val_dataset = splitter.train_test_split(dataset, frac_train=frac_train, seed=seed)
    train_df, val_df = split_dataframe(df, smiles_col, train_dataset, val_dataset)
    return train_df, val_df

def split_by_fingerprint(df, smiles_col, frac_train=0.75, seed=None):
    """Split the dataset by fingerprint."""
    dataset = smiles_to_deepchem_dataset(df, smiles_col)
    splitter = dc.splits.FingerprintSplitter()
    train_dataset, val_dataset = splitter.train_test_split(dataset, frac_train=frac_train, seed=seed)
    train_df, val_df = split_dataframe(df, smiles_col, train_dataset, val_dataset)
    return train_df, val_df

def split_randomly(df, smiles_col, frac_train=0.75, seed=None):
    """Split the dataset randomly."""
    dataset = smiles_to_deepchem_dataset(df, smiles_col)
    splitter = dc.splits.RandomSplitter()
    train_dataset, val_dataset = splitter.train_test_split(dataset, frac_train=frac_train, seed=seed)
    train_df, val_df = split_dataframe(df, smiles_col, train_dataset, val_dataset)
    return train_df, val_df

def get_scaffold(df, smiles_col):
    """Index the dataset by scaffold."""
    dataset = smiles_to_deepchem_dataset(df, smiles_col)
    splitter = dc.splits.ScaffoldSplitter()
    return splitter.generate_scaffolds(dataset), dataset

def get_scaffold_splits(domain_data, num_splits=5, random_seed=42):
    """
    Generate scaffold-based splits (train/test indices) for domain_data.
    This version assumes that the output of generate_scaffolds is a list of lists
    (each inner list contains indices of molecules sharing the same scaffold).
    """
    scaffolds, dataset = get_scaffold(domain_data, 'smiles')
    # Convert scaffold groups from indices to SMILES strings using dataset.ids.
    scaffold_groups = []
    for group in scaffolds:
        # For each scaffold group, gather the corresponding IDs (SMILES strings)
        group_ids = [dataset.ids[i] for i in group]
        scaffold_groups.append((group_ids, len(group_ids)))
    
    # Build a DataFrame where each row represents a scaffold group
    import pandas as pd
    scaffolds_df = pd.DataFrame(scaffold_groups, columns=["scaffold_group", "group_size"])

    splits = []
    for split_num in range(num_splits):
        # Create a reproducible random state for each split.
        rng = np.random.RandomState(random_seed + split_num)
        groups_shuffled = scaffold_groups.copy()
        rng.shuffle(groups_shuffled)

        train_groups, test_groups = [], []
        train_count, test_count = 0, 0
        for group, size in groups_shuffled:
            if train_count <= test_count:
                train_groups.append(group)
                train_count += size
            else:
                test_groups.append(group)
                test_count += size

        # Flatten the list of groups into one Series for train and test.
        import pandas as pd
        train_idx = pd.Series([item for group in train_groups for item in group])
        test_idx = pd.Series([item for group in test_groups for item in group])
        splits.append((train_idx, test_idx))
    
    return splits


def get_domain_data_split(domain_data, data_indices, idx_number):
    """Get the domain data splits according to provided indices."""
    if idx_number == 0:
        # For split 0, use scaffold splitting with a 50/50 partition.
        domain_data_train, domain_data_test = split_by_scaffold(domain_data[['label', 'smiles']], 'smiles', frac_train=0.50)
        domain_data_train = pd.merge(domain_data_train, domain_data, how='left', on='smiles')
        domain_data_test = pd.merge(domain_data_test, domain_data, how='left', on='smiles')
    else:
        # Here we assume that data_indices[idx_number] is a tuple (train_idx, test_idx)
        train_smi = data_indices[idx_number][0]  # Train SMILES from the tuple.
        test_smi  = data_indices[idx_number][1]  # Test SMILES from the tuple.
        domain_data_train = domain_data[domain_data['smiles'].isin(train_smi)].copy().reset_index(drop=True)
        domain_data_test = domain_data[domain_data['smiles'].isin(test_smi)].copy().reset_index(drop=True)
    return domain_data_train, domain_data_test

