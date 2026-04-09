import numpy as np
from matminer.datasets import load_dataset, get_all_dataset_info

def get_dataset_info(dataset_name: str = "matbench_dielectric") -> str:
    """Return dataset description from matminer."""
    return get_all_dataset_info(dataset_name)

def load_raw_dataset(dataset_name: str = "matbench_dielectric"):
    """Load raw dataset into a pandas DataFrame."""
    return load_dataset(dataset_name)

def prepare_dataframe(df, target_col: str = "n"):
    """
    Add composition column and log-transformed target.
    """
    df = df.copy()
    df["composition"] = df["structure"].apply(lambda x: x.composition)
    df[f"{target_col}_log"] = np.log(df[target_col])
    return df
