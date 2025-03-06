"""
Assertions and data inference functions
"""

from typing import Dict, Union, List, Tuple
from collections.abc import Callable

import pandas as pd
import numpy as np


def _prepare_everything(  # pylint: disable=too-many-arguments
    df: pd.DataFrame,
    clf: object,
    sample_id: str,
    feature_settings: Dict[str, Dict[str, Union[int, float, bool]]],
):

    # assert clf
    assert "predict_proba" in dir(clf), "clf must have a predict_proba method"
    assert (
        len(
            set(["feature_names_in_", "feature_name_", "feature_names"]).intersection(
                dir(clf)
            )
        )
        > 0
    ), "Could not find classifier's features"

    if "feature_names_in_" in dir(clf):  # logistic regression
        clf_features = clf.feature_names_in_
    elif "feature_name_" in dir(clf):  # LGBM
        clf_features = clf.feature_name_
    elif "feature_names" in dir(clf):  # XGboost
        clf_features = clf.feature_names
    else:
        clf_features = df.columns

    df = df.copy()

    # prepare sample_id
    df, sample_id_type = _handle_sample_id(df=df, sample_id=sample_id)

    # remove un-needed features
    df = df[clf_features]

    # infer continuous and categorical features
    if feature_settings:
        predefined_continuous = [
            k for k, v in feature_settings.items() if v.get("type") == "continuous"
        ]
        predefined_categorical = [
            k for k, v in feature_settings.items() if v.get("type") == "categorical"
        ]
    else:
        predefined_continuous = {}
        predefined_categorical = {}

    continuous_features = list(
        (set(_infer_continuous_features(df)) - set(predefined_categorical)).union(
            predefined_continuous
        )
    )
    print(f"Selected continuous features: {continuous_features}")

    categorical_features = list(
        (set(_infer_categorical_features(df)) - set(predefined_continuous)).union(
            predefined_categorical
        )
    )
    print(f"Selected categorical features: {categorical_features}")

    df = _assert_features(df, continuous_features, categorical_features)

    # assert feature_settings
    if feature_settings:
        _assert_feature_settings(df, feature_settings, continuous_features)

    return (df, sample_id_type, continuous_features, categorical_features)


def _handle_sample_id(
    df: pd.DataFrame, sample_id: str = None
) -> Tuple[pd.DataFrame, Callable, bool]:
    if sample_id:
        df.set_index(sample_id, inplace=True)

    # assert df's index is of a single type and has unique values
    assert len({type(c) for c in df.index}) == 1, "index must be of a single type"
    assert len(set(df.index)) == df.shape[0], "indices must be unique"

    # define index type
    sample_id_type = type(df.index[0])

    return df, sample_id_type


def _infer_continuous_features(df: pd.DataFrame) -> list:
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    continuous_features = list(df.select_dtypes(include=numerics).columns)
    continuous_features = [col for col in continuous_features if df[col].nunique() > 5]
    return continuous_features


def _infer_categorical_features(df: pd.DataFrame) -> list:
    categorical_features = [
        col
        for col in df.columns
        if df[col].dtype == "object"
        or df[col].dtype.name == "category"
        or df[col].nunique() <= 5
    ]
    return categorical_features


def _assert_features(
    df: pd.DataFrame, continuous_features: list, categorical_features: list
) -> pd.DataFrame:

    # assertions
    assert (
        len(set(continuous_features).intersection(categorical_features)) == 0
    ), f"Continuous and categorical features overlap: {list(set(continuous_features).intersection(categorical_features))}"
    assert (
        len(set(continuous_features) - set(df.columns)) == 0
    ), f"Some of continuous_features are not in dataframe: {list(set(continuous_features) - set(df.columns))}"
    assert (
        len(set(categorical_features) - set(df.columns)) == 0
    ), f"Some of categorical_features are not in dataframe: {list(set(categorical_features) - set(df.columns))}"
    assert all(
        pd.api.types.is_numeric_dtype(df[col]) for col in continuous_features
    ), "continuous_features include non-numeric columns"
    assert all(
        df[col].dtype == "object"
        or df[col].dtype.name == "category"
        or df[col].nunique() <= 10
        for col in categorical_features
    ), "categorical_features has non-categorical features ('object', 'category', or >10 unique values)"

    # remove columns from data frame that are not required
    redundant_features = (
        set(df.columns) - set(continuous_features) - set(categorical_features)
    )
    df.drop(list(redundant_features), axis=1, inplace=True)

    return df


def _assert_feature_settings(
    df: pd.DataFrame,
    feature_settings: Dict[str, Dict[str, Union[int, float]]],
    continuous_features: List[str],
) -> None:
    for k, v in feature_settings.items():
        arr = df[k]
        if k in df.columns and k in continuous_features:
            if "min" in v.keys():
                assert (
                    v["min"] <= arr.min()
                ), f"'min' slider value smaller than minimal value in {k}"
            if "max" in v.keys():
                assert (
                    v["max"] >= arr.max()
                ), f"'max' slider value greater than maximal value in {k}"
        if v.get("null") is False:
            assert (
                arr.isna().sum() == 0
            ), f"'null' was defined False for feature `{k}`, but column has NaN values"


def _calculate_min_max_value(arr: np.array, func: str, decimals: int):
    min_max = (np.min, -1) if func == "min" else (np.max, 1)
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]
    min_max_value = min_max[0](arr) + (np.std(arr) * min_max[1])
    return round(min_max_value * 1.0, decimals).astype(arr.dtype).item()


def _calculate_step_size(arr: pd.Series, decimals: int):
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]
    step_size = np.mean(np.diff(np.sort(np.unique(arr))))
    return (
        np.max([round(step_size, decimals), round(0.1, decimals)])
        .astype(arr.dtype)
        .item()
    )


def _get_sliders_params(arr, col_dict=None):
    col_dict = col_dict or dict()  # pylint: disable=use-dict-literal
    arr = np.array(arr)
    decimals = col_dict.get("decimals", 1)
    return {
        "type": "continuous",
        "min": col_dict.get(
            "min", _calculate_min_max_value(arr, func="min", decimals=decimals)
        ),
        "max": col_dict.get(
            "max", _calculate_min_max_value(arr, func="max", decimals=decimals)
        ),
        "step": col_dict.get("step", _calculate_step_size(arr, decimals)),
        "value": round(arr[0], decimals).astype(arr.dtype).item(),
        "null": col_dict.get("null", any(pd.isna(arr))),
    }


def _get_select_list_params(arr, col_dict=None):
    col_dict = col_dict or dict()  # pylint: disable=use-dict-literal
    options = arr.dropna().sort_values().astype(str).unique().tolist()
    allow_nulls = col_dict.get("null", any(pd.isna(arr)))
    value = arr.iloc[0]
    value = "" if pd.isna(value) else str(value)

    return {
        "type": "categorical",
        "options": [""] + options if allow_nulls else options,
        "value": value,
        "null": allow_nulls,
    }
