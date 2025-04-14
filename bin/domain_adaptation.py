# domain_adaptation.py

import argparse
import pickle as pkl
import pandas as pd
import numpy as np
from data_utils import get_scaffold_splits, get_domain_data_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import multiprocessing as mp

# Import our customized CatBoost wrapper and dataset config
from catboost_wrapper import CatBoostSklearnRegressor
from dataset_config import get_dataset_config

# Import existing adaptation methods.
from adapt.instance_based import TrAdaBoostR2, TwoStageTrAdaBoostR2, NearestNeighborsWeighting, KMM, KLIEP

# --- New Adaptation Adapter Classes ---

class CombinedDataAdapter:
    """
    A simple adapter that trains on the combined data
    (source data + domain training data).
    """
    def __init__(self, reg, **kwargs):
        self.reg = reg

    def fit(self, source_X, source_y, Xt, yt):
        # Combine source and domain training data.
        X_combined = np.concatenate([source_X, Xt], axis=0)
        y_combined = np.concatenate([source_y, yt], axis=0)
        self.reg.fit(X_combined, y_combined)
        return self

    def predict(self, X):
        return self.reg.predict(X)

class ContinuedTrainingAdapter:
    """
    An adapter that first trains on the source data and then
    continues training on the domain data.
    """
    def __init__(self, reg, **kwargs):
        self.reg = reg
        # Additional iterations for domain-specific training.
        self.additional_iterations = kwargs.get("additional_iterations", 1000)

    def fit(self, source_X, source_y, Xt, yt):
        # First train on source data.
        self.reg.fit(source_X, source_y)
        # Then continue training on domain training data.
        # This uses the 'init_model' parameter of CatBoost to continue training.
        self.reg.model.fit(
            Xt, yt,
            init_model=self.reg.model,
            verbose=False,
            iterations=self.additional_iterations
        )
        return self

    def predict(self, X):
        return self.reg.predict(X)

# --- Adapter Factory ---

def get_adapter_instance(method_name, reg, domain_train_X, domain_train_y, adapter_params):
    """
    Create an adaptation adapter instance based on the provided method name.
    For 'KMM' and 'KLIEP', Xt and yt are omitted at initialization.
    """
    adapter_map = {
        "TrAdaBoostR2": TrAdaBoostR2,
        "TwoStageTrAdaBoostR2": TwoStageTrAdaBoostR2,
        "NearestNeighborsWeighting": NearestNeighborsWeighting,
        "KMM": KMM,
        "KLIEP": KLIEP,
        "CombinedDataAdapter": CombinedDataAdapter,
        "ContinuedTrainingAdapter": ContinuedTrainingAdapter,
    }
    AdapterClass = adapter_map.get(method_name)
    if AdapterClass is None:
        raise ValueError(f"Adapter method {method_name} not recognized.")

    # For KMM and KLIEP, do not pass Xt and yt to the constructor.
    if method_name in ["KMM", "KLIEP"]:
        adapter = AdapterClass(reg, **adapter_params)
    else:
        adapter = AdapterClass(reg, **adapter_params)
    return adapter

# --- Training Task ---

def train_fraction(task, dataset_config, adaptation_method, adapter_params, num_fractions):
    """
    Train and evaluate the adapted model for a given scaffold split and fraction of data.
    
    task: tuple (split_idx, fraction_index, domain_train, domain_test, source_data)
    """
    split_idx, fraction_index, domain_train, domain_test, source_data = task

    # Determine fraction of training data to use.
    total_samples = len(domain_train)
    x = int(fraction_index * total_samples / num_fractions)
    fraction_included = x / total_samples

    # Unpack relevant column names from dataset config.
    features_col = dataset_config.get("features_col", "comb_features")
    domain_target_col = dataset_config.get("domain_target")
    source_target_col = dataset_config.get("source_target")

    # Two-stage scaling: (1) fit scaler on a fraction of domain_train, then (2) overwrite using source_data.
    scaler = RobustScaler()

    # Fit on fraction of domain training set.
    domain_train_features = domain_train[features_col].tolist()[:x]
    domain_train_X = scaler.fit_transform(domain_train_features)
    domain_train_y = np.array(domain_train[domain_target_col].tolist()[:x])
    
    # Refit on entire source data.
    source_features = source_data[features_col].tolist()
    source_X = scaler.fit_transform(source_features)
    source_y = np.array(source_data[source_target_col].tolist())

    # Instantiate the regression model using dataset-specific parameters.
    reg = CatBoostSklearnRegressor(dataset_config.get("dataset_name"), **dataset_config.get("catboost_params", {}))

    # Create adapter instance based on the selected adaptation method.
    adapter = get_adapter_instance(adaptation_method, reg, domain_train_X, domain_train_y, adapter_params)
    # Fit the adapter using source data and domain training data.
    adapter.fit(source_X, source_y, Xt=domain_train_X, yt=domain_train_y)

    # Prepare the test data.
    test_features = domain_test[features_col].tolist()
    domain_test_X = scaler.transform(test_features)
    domain_test_y = np.array(domain_test[domain_target_col].tolist())
    predictions = adapter.predict(domain_test_X)
    RMSE = mean_squared_error(domain_test_y, predictions)

    result = {
        "split_idx": split_idx,
        "fraction_index": fraction_index,
        "fraction_included": fraction_included,
        "source_size": source_y.shape[0],
        "RMSE": RMSE,
        "adapter_method": adaptation_method
    }
    return result

# --- Argument Parsing ---

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Domain adaptation with configurable dataset and method"
    )
    parser.add_argument("--source_data_path", type=str,
                        default="/home/ubuntu/bio2d_public/data/clean_nih_sol_fp.pkl",
                        help="Path to the source dataset pickle file")
    parser.add_argument("--domain_data_path", type=str,
                        default="/home/ubuntu/bio2d_public/data/clean_biogen_sol_fp.pkl",
                        help="Path to the domain dataset pickle file")
    parser.add_argument("--output_path", type=str,
                        default="/home/ubuntu/bio2d_public/data/solubility_metrics.csv",
                        help="CSV output file for evaluation metrics")
    parser.add_argument("--dataset_name", type=str, default="solubility",
                        help="Dataset name used to load configuration (e.g., solubility, hppb, hlm)")
    parser.add_argument("--adaptation_method", type=str, default="KMM",
                        help="Adaptation method. Options: TrAdaBoostR2, TwoStageTrAdaBoostR2, "
                             "NearestNeighborsWeighting, KMM, KLIEP, CombinedDataAdapter, ContinuedTrainingAdapter")
    parser.add_argument("--num_fractions", type=int, default=10,
                        help="Number of fractions into which the domain training set is divided")
    parser.add_argument("--num_scaffold_splits", type=int, default=5,
                        help="Number of scaffold splits to generate from the domain dataset")
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Number of processes for parallel training")
    # Adapter-specific hyperparameters.
    parser.add_argument("--adapter_n_estimators", type=int, default=10,
                        help="Number of estimators for the adaptation method")
    parser.add_argument("--adapter_random_state", type=int, default=0,
                        help="Random seed for the adaptation method")
    # For ContinuedTrainingAdapter: additional iterations for domain training.
    parser.add_argument("--additional_iterations", type=int, default=1000,
                        help="Additional iterations when continuing training on domain data")
    return parser.parse_args()

# --- Main ---

if __name__ == "__main__":
    args = get_args()
    
    # Load the dataset config for the given dataset name.
    dataset_config = get_dataset_config(args.dataset_name)
    # Also add the dataset name to the config for later use.
    dataset_config["dataset_name"] = args.dataset_name

    # Load source and domain data.
    source_data = pd.read_pickle(args.source_data_path)
    domain_data = pd.read_pickle(args.domain_data_path)

    # Remove overlapping molecules based on SMILES.
    overlap = domain_data['smiles'].isin(source_data['smiles'])
    domain_data = domain_data[~overlap].copy()

    # Create scaffold splits.
    scaffold_indices = get_scaffold_splits(domain_data, num_splits=args.num_scaffold_splits)
    scaffold_splits = []
    for i in range(args.num_scaffold_splits):
        domain_train, domain_test = get_domain_data_split(domain_data, scaffold_indices, i)
        scaffold_splits.append((domain_train, domain_test))

    # Build task list: each task corresponds to one scaffold split and one training fraction.
    tasks = []
    for split_idx, (domain_train, domain_test) in enumerate(scaffold_splits):
        for fraction_index in range(1, args.num_fractions + 1):
            tasks.append((split_idx, fraction_index, domain_train, domain_test, source_data))

    # Parameters to pass to the adapter.
    adapter_params = {
        "n_estimators": args.adapter_n_estimators,
        "random_state": args.adapter_random_state,
        "additional_iterations": args.additional_iterations,  # for ContinuedTrainingAdapter
    }
    
    # Multiprocessing: use a helper lambda to pass extra arguments.
    pool = mp.Pool(processes=args.num_processes)
    func = lambda t: train_fraction(t, dataset_config, args.adaptation_method, adapter_params, args.num_fractions)
    results_list = pool.map(func, tasks)
    pool.close()
    pool.join()

    # Aggregate results and save to CSV.
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(args.output_path, index=False)
