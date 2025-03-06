import unittest
from string import ascii_letters

import pandas as pd
import numpy as np


from whatifact._inference import (
    _calculate_step_size,
    _calculate_min_max_value,
    _handle_sample_id,
    _get_sliders_params,
    _infer_continuous_features,
    _infer_categorical_features,
    _assert_features,
    _assert_feature_settings,
)


class TestAssertions(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "col1": [1, 2, 3] * 10,  # categorical, nunique() < 5
                "col2": list(ascii_letters[:30]),  # categorical, type object
                "col3": [i for i in range(10)]
                * 3,  # categorical, will be defined below
                "col4": np.arange(30),  # continuous, int
                "col5": np.random.normal(size=30),  # continuous, float
            }
        )

        self.df["col3"] = self.df["col3"].astype("category")
        self.categorical_features = ["col1", "col2", "col3"]
        self.continuous_features = ["col4", "col5"]

    def test_infer_continuous_features(self):
        continuous_features = set(_infer_continuous_features(self.df))
        self.assertEqual(continuous_features, {"col4", "col5"})

    def test_infer_categorical_features(self):
        categorical_features = set(_infer_categorical_features(self.df))
        self.assertEqual(categorical_features, {"col1", "col2", "col3"})

    def test_assert_features_pass(self):
        _ = _assert_features(
            df=self.df,
            categorical_features=self.categorical_features,
            continuous_features=self.continuous_features,
        )

    def test_assert_features_fail_overlap(self):
        with self.assertRaises(AssertionError) as e:
            _ = _assert_features(
                df=self.df,
                categorical_features=self.categorical_features,
                continuous_features=self.continuous_features
                + [self.categorical_features[0]],
            )
        self.assertIn("Continuous and categorical features overlap", str(e.exception))

    def test_assert_features_fail_unavailable_continuous_features(self):
        with self.assertRaises(AssertionError) as e:
            _ = _assert_features(
                df=self.df,
                categorical_features=self.categorical_features,
                continuous_features=self.continuous_features + ["col9"],
            )
        self.assertIn(
            "Some of continuous_features are not in dataframe", str(e.exception)
        )

    def test_assert_features_fail_unavailable_categorical_features(self):
        with self.assertRaises(AssertionError) as e:
            _ = _assert_features(
                df=self.df,
                categorical_features=self.categorical_features + ["col9"],
                continuous_features=self.continuous_features,
            )
        self.assertIn(
            "Some of categorical_features are not in dataframe", str(e.exception)
        )

    def test_assert_features_fail_wrong_continuous_features(self):
        with self.assertRaises(AssertionError) as e:
            _ = _assert_features(
                df=self.df,
                categorical_features=self.categorical_features[:2],
                continuous_features=self.continuous_features
                + [self.categorical_features[-1]],
            )
        self.assertIn(
            "continuous_features include non-numeric columns", str(e.exception)
        )

    def test_assert_features_fail_wrong_categorical_features(self):
        with self.assertRaises(AssertionError) as e:
            _ = _assert_features(
                df=self.df,
                categorical_features=self.categorical_features
                + [self.continuous_features[-1]],
                continuous_features=self.continuous_features[:1],
            )
        self.assertIn(
            "categorical_features has non-categorical features", str(e.exception)
        )

    def test_assert_features_drop_columns(self):
        df_redundant = self.df.copy().assign(col6=np.arange(30))

        df_asserted = _assert_features(
            df_redundant,
            categorical_features=self.categorical_features,
            continuous_features=self.continuous_features,
        )
        self.assertEqual(set(self.df.columns), set(df_asserted.columns))

    def test_assert_feature_settings_pass(self):
        feature_settings = {"col4": {"min": -5, "max": 100, "step": 2, "decimals": 3}}
        _assert_feature_settings(
            df=self.df,
            feature_settings=feature_settings,
            continuous_features=self.continuous_features,
        )

    def test_assert_feature_settings_min_too_high(self):
        feature_settings = {"col4": {"min": 1, "max": 100, "step": 2, "decimals": 3}}
        with self.assertRaises(AssertionError) as e:
            _assert_feature_settings(
                df=self.df,
                feature_settings=feature_settings,
                continuous_features=self.continuous_features,
            )
        self.assertIn("'min' slider value smaller", str(e.exception))

    def test_assert_feature_settings_max_too_low(self):
        feature_settings = {"col4": {"min": -1, "max": 20, "step": 2, "decimals": 3}}
        with self.assertRaises(AssertionError) as e:
            _assert_feature_settings(
                df=self.df,
                feature_settings=feature_settings,
                continuous_features=self.continuous_features,
            )
        self.assertIn("'max' slider value greater", str(e.exception))


class TestHandleSampleID(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "sample_id": np.arange(30) + 3000,
                "col1": [1, 2, 3] * 10,  # categorical, nunique() < 5
                "col2": list(ascii_letters[:30]),  # categorical, type object
                "col3": [i for i in range(10)]
                * 3,  # categorical, will be defined below
                "col4": np.arange(30),  # continuous, int
                "col5": np.random.normal(size=30),  # continuous, float
            }
        )
        self.categorical_features = ["col1", "col2", "col3"]
        self.continuous_features = ["col4", "col5"]

    def test_handle_sample_id_pass(self):
        df, sample_id_type = _handle_sample_id(df=self.df, sample_id="sample_id")

        self.assertEqual(len(df.columns), 5)
        self.assertEqual(df.index.name, "sample_id")
        self.assertEqual(sample_id_type("3"), 3)

    def test_handle_sample_id_no_sample_id(self):
        df, sample_id_type = _handle_sample_id(df=self.df)

        self.assertEqual(len(df.columns), 6)
        self.assertEqual(df.index.name, None)
        self.assertEqual(sample_id_type("3"), 3)

    def test_handle_sample_id_string_sample_id(self):
        df, sample_id_type = _handle_sample_id(df=self.df, sample_id="col2")

        self.assertEqual(len(df.columns), 5)
        self.assertEqual(df.index.name, "col2")
        self.assertEqual(sample_id_type(3), "3")

    def test_handle_sample_id_no_multiple_sample_id_dtype(self):
        df = self.df.copy()
        df["new_sample_id"] = [i for i in range(15)] + list(ascii_letters[:15])

        with self.assertRaises(AssertionError) as e:
            df, sample_id_type = _handle_sample_id(df=df, sample_id="new_sample_id")
        self.assertIn("index must be of a single type", str(e.exception))

    def test_handle_sample_id_non_unique_sample_id(self):
        df = self.df.copy()
        df.loc[0, "sample_id"] = df.loc[1, "sample_id"]

        with self.assertRaises(AssertionError) as e:
            df, sample_id_type = _handle_sample_id(df=df, sample_id="sample_id")
        self.assertIn("indices must be unique", str(e.exception))


class TestInference(unittest.TestCase):
    def setUp(self):
        self.arr1 = np.linspace(-5, 5, 100)
        self.arr2 = np.arange(10)

        mask = np.random.binomial(p=0.1, n=1, size=self.arr1.shape)
        self.arr3 = self.arr1.copy()
        self.arr3[mask] = np.nan

        self.v = dict(id="id", caption="caption", min=1, max=10, value=3, step=1)

    def test_calculate_min_max_value(self):
        r = _calculate_min_max_value(self.arr1, func="min", decimals=1)
        self.assertLess(r, self.arr1.min())

        r = _calculate_min_max_value(self.arr1, func="max", decimals=1)
        self.assertGreater(r, self.arr1.max())
        self.assertIsInstance(r, float)

        r = _calculate_min_max_value(self.arr2, func="max", decimals=1)
        self.assertGreater(r, self.arr1.max())
        self.assertIsInstance(r, int)

    def test_calculate_min_max_value_with_missing(self):
        r = _calculate_min_max_value(self.arr3, func="min", decimals=1)
        self.assertLess(r, np.nanmin(self.arr3))
        self.assertIsInstance(r, float)

    def test_calculate_step_size(self):
        r = _calculate_step_size(self.arr2, decimals=1)
        self.assertEqual(r, 1)

    def test_calculate_step_size_with_missing(self):
        r = _calculate_step_size(self.arr3, decimals=1)
        self.assertIsInstance(r, float)

    def test_calculate_step_size(self):
        r = _calculate_step_size(self.arr1, decimals=3)
        self.assertIsInstance(r, float)

    def test_get_sliders_params(self):
        r = _get_sliders_params(self.arr2)

        self.assertIsInstance(r, dict)
        self.assertEqual(set(r.keys()), {"min", "max", "step", "value", "null", "type"})
        self.assertEqual(r["value"], 0.0)


if __name__ == "__main__":
    unittest.main()
