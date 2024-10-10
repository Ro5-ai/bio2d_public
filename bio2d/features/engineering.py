import numpy as np
import pandas as pd

from imblearn.over_sampling import SVMSMOTE
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

from copy import deepcopy

from abc import ABC, abstractmethod

SCALABLE_FEATURES = ['rdkit_desc', 'erg']


class BaseFeatureTransformer(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        """
        pass

    @abstractmethod
    def transform(self, X):
        """
        Apply the model to the data X.
        """
        pass

    def fit_transform(self, X, y=None):
        """
        Fit the model according to the given training data, then transform it.
        """
        self.fit(X, y)
        return self.transform(X)


class CorrelatedFingerprintCounts(BaseFeatureTransformer):
    """
    Method based on LanternPharma's approach:
    https://github.com/lanternpharma/tdc-bbb-martins/blob/main/code/logistic_fe_aug.py
    """

    def __init__(self, binary_feature_list=[]):
        self.neg_sig_fingerprints = None
        self.pos_sig_fingerprints = None
        self.fingerprint_features = None

        if binary_feature_list == []:
            self.binary_feature_list = self.get_default_binary_features()

    def get_default_binary_features(self):
        return ["maccs_keys", "circular", "atom_pair", "rdkit_fp"]

    def fit(self, X, y):
        tmp_df = deepcopy(X)
        tmp_df['target'] = y

        binary_cols = [col for col in X.columns if any(col.startswith(prefix) for prefix in self.binary_feature_list)]

        fp_df = pd.DataFrame(list(binary_cols), columns=['Feature'])

        fp_percent_permeable = []
        for i in binary_cols:
            if tmp_df[i].max() == 0:
                fp_percent_permeable.append(np.nan)
            else:
                fp_percent_permeable.append(tmp_df.groupby(i)['target'].mean()[1])

        fp_df['percent_permeable'] = fp_percent_permeable

        self.neg_sig_fingerprints = list(fp_df[(fp_df.percent_permeable < 0.40)].Feature)
        self.pos_sig_fingerprints = list(fp_df[(fp_df.percent_permeable > 0.80)].Feature)

    def transform(self, X, y):
        if self.neg_sig_fingerprints is None or self.pos_sig_fingerprints is None:
            raise Exception("The model has not been fit yet. Please call fit() before transform().")

        X['neg_sig_fingerprints'] = X[self.neg_sig_fingerprints].sum(axis=1)
        X['pos_sig_fingerprints'] = X[self.pos_sig_fingerprints].sum(axis=1)
        return X, y


class SmoteUpsampler(BaseFeatureTransformer):
    def __init__(self):
        self.upsampler = SVMSMOTE(random_state=42)

    def fit(self, X, y):
        pass

    def transform(self, X, y):
        cols = X.columns
        X = X.to_numpy()
        new_X, new_y = self.upsampler.fit_resample(X, y)
        new_df = pd.DataFrame(new_X, columns=cols)
        return new_df, new_y


class FeatureScaler(BaseFeatureTransformer):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        scale_cols = [col for col in X.columns if any(col.startswith(prefix) for prefix in SCALABLE_FEATURES)]
        if len(scale_cols) == 0:
            self.fit_flag = False
        else:
            self.fit_flag = True
            self.scaler.fit(X[scale_cols].to_numpy())

    def transform(self, X, y=None):
        if not self.fit_flag:
            return X, y
        scale_cols = [col for col in X.columns if any(col.startswith(prefix) for prefix in SCALABLE_FEATURES)]
        scaled_data = self.scaler.transform(X[scale_cols].to_numpy())
        scaled_df = pd.DataFrame(scaled_data, columns=scale_cols)
        reduced_df = X.drop(scale_cols, axis=1)
        result_df = pd.concat([reduced_df, scaled_df], axis=1)
        return result_df, y


class FeatureSelector(BaseFeatureTransformer):
    """
    Method based on LanternPharma's approach:
    https://github.com/lanternpharma/tdc-bbb-martins/blob/main/code/logistic_fe_aug.py
    """

    def __init__(self):
        self.model = None
        self.selector = None

    def values_are_boolean(self, target_values):
        return all(i == 0 or i == 1 for i in target_values)

    def fit(self, X, y):
        if self.values_are_boolean(y):
            model_class = LogisticRegression
            best_params = {'C': 1.00, 'class_weight': 'balanced', 'random_state': 8516, 'solver': 'liblinear'}
            self.model = model_class(**best_params, penalty='l1')
            self.model.fit(X.to_numpy(), y)
            coefs = self.model.coef_[0]
        else:
            model_class = Lasso
            self.model = model_class()
            self.model.fit(X.to_numpy(), y)
            coefs = self.model.coef_

        keep_idx = []
        for ii, coef in enumerate(coefs):
            if coef != 0:
                keep_idx.append(ii)

        self.selected_cols = [X.columns[ii] for ii in keep_idx]

    def transform(self, X, y):
        return X[self.selected_cols], y
