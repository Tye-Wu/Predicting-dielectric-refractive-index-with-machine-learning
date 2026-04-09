import numpy as np
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.structure.order import DensityFeatures

def featurize_dataframe(df):
    """
    Add composition- and structure-based descriptors.
    """
    df = df.copy()

    el_prop = ElementProperty.from_preset("magpie")
    el_prop.set_n_jobs(1)
    df = el_prop.featurize_dataframe(df, col_id="composition", ignore_errors=True)

    density = DensityFeatures()
    df = density.featurize_dataframe(df, col_id="structure", ignore_errors=True)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def build_feature_matrix(df, target_col: str = "n_log"):
    """
    Build X, y and return feature column names.
    """
    features_to_drop = ["structure", "composition", "n", "n_log"]
    feature_cols = [c for c in df.columns if c not in features_to_drop]

    X = df[feature_cols].values
    y = df[target_col].values
    return X, y, feature_cols
