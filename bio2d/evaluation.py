from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, average_precision_score, mean_absolute_error

import numpy as np
from scipy.stats import pearsonr, spearmanr


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def scorer(y_pred, y_true, task_type, bool_threshold=0.5, iqr=None):
    if task_type == "binary":
        # convert probabilities to bool

        y_bool_pred = [0 if y < bool_threshold else 1 for y in y_pred]

        result = {
            "acc": accuracy_score(y_true, y_bool_pred),
            "f1": f1_score(y_true, y_bool_pred),
            "recall": recall_score(y_true, y_bool_pred),
            "precision": precision_score(y_true, y_bool_pred),
            "rocauc": roc_auc_score(y_true, y_pred, average='weighted'),
            "auprc": average_precision_score(y_true, y_pred),
        }

        result = {k:round(v,3) for k,v in result.items()}
        return result
    elif task_type == "regression":
        nrmse = None
        if iqr is not None:
            nrmse = round(rmse(y_true, y_pred) / iqr, 3)

        result = {
            "pearson": pearsonr(y_true, y_pred)[0],
            "spearman": spearmanr(y_true, y_pred)[0],
            "rmse": rmse(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "nrmse": nrmse,
        }

        result = {k:round(v,3) for k,v in result.items()}
        return result
