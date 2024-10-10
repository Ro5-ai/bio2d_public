from dataclasses import dataclass, field

from typing import List

from bio2d.features.engineering import FeatureSelector, CorrelatedFingerprintCounts, SmoteUpsampler, FeatureScaler, BaseFeatureTransformer


@dataclass
class FeatureManipulationMethods:
    feature_list: List[str]
    scaling: BaseFeatureTransformer = field(default_factory=lambda: None)
    upsampling: BaseFeatureTransformer = field(default_factory=lambda: None)
    correlated_fingerprint_counts: BaseFeatureTransformer = field(default_factory=lambda: None)
    feature_selection: BaseFeatureTransformer = field(default_factory=lambda: None)


def get_feature_method(feature_type):
    try:
        return FEATURE_TYPE_TO_METHOD[feature_type]
    except KeyError:
        raise ValueError(f"No variable defined for feature type {feature_type}")


def get_scaling_config(feature_list):
    fmm = FeatureManipulationMethods(feature_list=feature_list, scaling=FeatureScaler)
    return fmm


BASE_FEATURES = ['rdkit_desc', 'ecfp4', 'avalon', 'erg']

BASE_SET = FeatureManipulationMethods(feature_list=BASE_FEATURES)

BASE_WITH_SCALING = FeatureManipulationMethods(feature_list=BASE_FEATURES, scaling=FeatureScaler)

BASE_WITH_UPSAMPLING = FeatureManipulationMethods(feature_list=BASE_FEATURES, upsampling=SmoteUpsampler)

BASE_WITH_ENGINEERING = FeatureManipulationMethods(feature_list=BASE_FEATURES, correlated_fingerprint_counts=CorrelatedFingerprintCounts)

BASE_WITH_SELECTION = FeatureManipulationMethods(feature_list=BASE_FEATURES, feature_selection=FeatureSelector)

BASE_WITH_ALL = FeatureManipulationMethods(
    feature_list=BASE_FEATURES,
    scaling=FeatureScaler,
    upsampling=SmoteUpsampler,
    correlated_fingerprint_counts=CorrelatedFingerprintCounts,
    feature_selection=FeatureSelector,
)


FEATURE_TYPE_TO_METHOD = {
    'base_set': BASE_SET,
    'base_with_scaling': BASE_WITH_SCALING,
    'base_with_selection': BASE_WITH_SELECTION,
    'base_with_engineering': BASE_WITH_ENGINEERING,
    'base_with_upsampling': BASE_WITH_UPSAMPLING,
    'base_with_all': BASE_WITH_ALL,
}
