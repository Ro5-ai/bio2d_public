# dataset_config.py

DATASET_CONFIG = {
    "solubility": {
        "source_target": "nih_log(mol/L)",
        "domain_target": "biogen_log(mol/L)",
        "features_col": "comb_features",
        "catboost_params": {
            "depth": 4,
            "learning_rate": 0.05,
            "iterations": 2000,
            "bagging_temperature": 5,
            "l2_leaf_reg": 3,
            "verbose": 1000,
        }
    },
    "hppb": {
        "source_target": "az_log_%_unbound",
        "domain_target": "biogen_log_%_unbound",
        "features_col": "comb_features",
        "catboost_params": {
            "depth": 4,
            "learning_rate": 0.05,
            "iterations": 2000,
            "bagging_temperature": 5,
            "l2_leaf_reg": 3,
            "verbose": 1000,
        }
    },
    "hlm": {
        "source_target": "az_log_mL/min/g",
        "domain_target": "biogen_log_mL/min/kg",
        "features_col": "comb_features",
        "catboost_params": {
            "depth": 4,
            "learning_rate": 0.05,
            "iterations": 2000,
            "bagging_temperature": 5,
            "l2_leaf_reg": 3,
            "verbose": 1000,
        }
    }
}

def get_dataset_config(dataset_name):
    return DATASET_CONFIG.get(dataset_name, {})

