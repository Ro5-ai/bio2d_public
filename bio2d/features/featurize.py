import deepchem
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdReducedGraphs
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

from bio2d.features.hermes_code import smile_to_fingerprint

from typing import List
import pandas as pd


class FeatureBuilder:
    """
    A class that provides methods for building various types of molecular descriptors or fingerprints for a list of SMILES strings.
    """

    def __init__(self):
        pass

    def get_rdkit_fp(self, smiles_list: List[str]):
        """
        Computes RDKit fingerprints for a list of SMILES strings.

        Args:
            smiles_list (List[str]): List of SMILES strings to be featurized.

        Returns:
            np.ndarray: Array containing the computed fingerprints for each SMILES string.
        """

        fps = [smile_to_fingerprint(smiles, fingerprint_type='rdkit') for smiles in smiles_list]
        return np.array(fps)

    def get_autocorr_2d(self, smiles_list: List[str]):
        """
        Computes 2D autocorrelations for a list of SMILES strings.

        Args:
            smiles_list (List[str]): List of SMILES strings to be featurized.

        Returns:
            np.ndarray: Array containing the computed 2D autocorrelations for each SMILES string.
        """

        mol_list = [Chem.MolFromSmiles(sm) for sm in smiles_list]
        fps = [Chem.rdMolDescriptors.CalcAUTOCORR2D(mol) for mol in mol_list]
        return np.array(fps)

    def get_maccs_keys(self, smiles_list: List[str]):
        """
        Computes MACCS keys fingerprints for a list of SMILES strings.

        Args:
            smiles_list (list of str):
                List of SMILES strings to be featurized.

        Returns:
            features (numpy array, shape (n_samples, n_features)):
                Array containing the computed features for each SMILES string.

        """

        mol_list = [Chem.MolFromSmiles(sm) for sm in smiles_list]
        ffs = [rdMolDescriptors.GetMACCSKeysFingerprint(mol) for mol in mol_list]

        return np.array(ffs)

    def get_ecfp4(self, smiles_list: List[str]):
        """
        Computes ECFP4 fingerprints for a list of SMILES strings.

        Args:
            smiles_list (list of str):
                List of SMILES strings to be featurized.

        Returns:
            features (numpy array, shape (n_samples, n_features)):
                Array containing the computed features for each SMILES string.

        """
        fps = [smile_to_fingerprint(smiles, fingerprint_type='morgan') for smiles in smiles_list]
        return np.array(fps)

    def get_mol2vec(self, smiles_list: List[str]):
        """
        Computes Mol2Vec fingerprints for a list of SMILES strings.

        Args:
            smiles_list (list of str):
                List of SMILES strings to be featurized.

        Returns:
            features (numpy array, shape (n_samples, n_features)):
                Array containing the computed features for each SMILES string.

        """
        featurizer = deepchem.feat.Mol2VecFingerprint()
        features = featurizer.featurize(smiles_list)
        return np.array(features)

    def get_mordred(self, smiles_list: List[str]):
        """
        Computes Mordred descriptors for a list of SMILES strings.

        Args:
            smiles_list (list of str):
                List of SMILES strings to be featurized.

        Returns:
            features (numpy array, shape (n_samples, n_features)):
                Array containing the computed features for each SMILES string.

        """
        featurizer = deepchem.feat.MordredDescriptors(ignore_3D=True)
        features = featurizer.featurize(smiles_list)
        return np.array(features)

    def get_rdkit_desc(self, smiles_list: List[str]):
        """
        Computes RDKit descriptors for a list of SMILES strings.

        Args:
            smiles_list (list of str):
                List of SMILES strings to be featurized.

        Returns:
            features (numpy array, shape (n_samples, n_features)):
                Array containing the computed features for each SMILES string.

        """
        featurizer = deepchem.feat.RDKitDescriptors()
        features = featurizer.featurize(smiles_list)
        return np.array(features)

    def get_pubchem(self, smiles_list: List[str]):
        """
        Computes PubChem fingerprints for a list of SMILES strings.

        Args:
            smiles_list (list of str):
                List of SMILES strings to be featurized.

        Returns:
            features (numpy array, shape (n_samples, n_features)):
                Array containing the computed features for each SMILES string.

        """
        featurizer = deepchem.feat.PubChemFingerprint()
        features = featurizer.featurize(smiles_list)
        return np.array(features)

    def get_atom_pair(self, smiles_list: List[str]):
        """
        Computes AtomPair fingerprints for a list of SMILES strings.

        Args:
            smiles_list (list of str):
                List of SMILES strings to be featurized.

        Returns:
            features (numpy array, shape (n_samples, n_features)):
                Array containing the computed features for each SMILES string.

        """
        mols = [Chem.MolFromSmiles(sm) for sm in smiles_list]
        features = [Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol) for mol in mols]
        return np.array(features)

    def get_erg(self, smiles_list: List[str]):
        """
        Computes ErG fingerprints for a list of SMILES strings.

        Args:
            smiles_list (List[str]): List of SMILES strings to be featurized.

        Returns:
            np.ndarray: Array containing the computed ErG fingerprints for each SMILES string.
        """
        mol_list = [Chem.MolFromSmiles(sm) for sm in smiles_list]
        fps = [rdReducedGraphs.GetErGFingerprint(mol) for mol in mol_list]
        return np.array(fps)

    def get_avalon(self, smiles_list: List[str], n_bits=1024):
        """
        Computes Avalon fingerprints for a list of SMILES strings.

        Args:
            smiles_list (List[str]): List of SMILES strings to be featurized.
            n_bits (int): The number of bits for the fingerprint.

        Returns:
            np.ndarray: Array containing the computed Avalon fingerprints for each SMILES string.
        """
        mol_list = [Chem.MolFromSmiles(sm) for sm in smiles_list]
        fps = [GetAvalonFP(mol, nBits=n_bits) for mol in mol_list]
        return np.array(fps)


def features_from_smiles(smiles_list: List[str], features_list: List[str], s3_uri: str = None, test: bool = False):
    """Extracts features from SMILES strings using DeepChem featurizers.

    Args:
        smiles_list (list of str): A list of SMILES strings.
        features_list (list of str, optional): A list of feature types to extract. Defaults to None, in which case
        all available feature types will be extracted.


    Returns:
        dict: A dictionary of feature arrays, with the keys being the names of the featurizers.
    """
    features = {}
    for feature_type in features_list:
        fb = FeatureBuilder()
        ff = getattr(fb, f"get_{feature_type}")(smiles_list)
        ff = np.nan_to_num(ff, nan=0, posinf=0)  # convert nan to 0
        features[feature_type] = ff
    df = combine_features(features)
    return df


def combine_features(features: dict) -> pd.DataFrame:
    """
    Combines features from different featurizers into a single DataFrame.

    Args:
        features (dict): Dictionary containing the feature arrays, with keys being the names of the featurizers.

    Returns:
        pd.DataFrame: DataFrame containing the combined features.
    """
    df = pd.DataFrame()
    for feature_type, feature_values in features.items():
        feature_dimensionality = feature_values.shape[1]
        column_names = [f'{feature_type}_{idx}' for idx in range(feature_dimensionality)]
        temp_df = pd.DataFrame(feature_values, columns=column_names)
        df = pd.concat([df, temp_df], axis=1)
    return df
