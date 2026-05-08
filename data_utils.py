import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = "data_23k_v1.csv"

TABULAR_FEATURES = [
    "Correction_Length", "Correction_Deletion", "Correction_Insertion",
    "Correction_Replacement", "RToverhangmatches", "RToverhanglength",
    "RTlength", "PBSlength", "RTmt", "RToverhangmt", "PBSmt",
    "protospacermt", "extensionmt", "original_base_mt", "edited_base_mt",
    "original_base_mt_nan", "edited_base_mt_nan", "deepeditposition",
    "Editing_Position",
]

TARGET_HEK = "HEKaverageedited_clamped"
TARGET_K562 = "K562averageedited_clamped"

CORRECTION_TYPE_MAP = {"Replacement": 0, "Insertion": 1, "Deletion": 2}


def load_df(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["correction_type_enc"] = df["Correction_Type"].map(CORRECTION_TYPE_MAP)
    return df


def get_sequences(df: pd.DataFrame) -> list[str]:
    return df["wide_mutated_target"].tolist()


def group_split(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1, seed: int = 42):
    groups = df["grp_id"].values
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(df, groups=groups))

    df_trainval = df.iloc[trainval_idx]
    groups_trainval = groups[trainval_idx]

    val_frac = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_rel_idx, val_rel_idx = next(gss_val.split(df_trainval, groups=groups_trainval))

    train_idx = trainval_idx[train_rel_idx]
    val_idx = trainval_idx[val_rel_idx]

    return train_idx, val_idx, test_idx


def build_tabular(df: pd.DataFrame) -> np.ndarray:
    cols = TABULAR_FEATURES + ["correction_type_enc"]
    return df[cols].values.astype(np.float32)