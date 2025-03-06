import unittest
from string import ascii_letters

import numpy as np
import pandas as pd

from whatifact._utilities import _get_variables_and_widgets, _update_values


class TestPreprocessing(unittest.TestCase):
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

    def test_get_variables_and_widgets(self):
        variables, widgets = _get_variables_and_widgets(
            df=self.df.set_index("sample_id"),
            continuous_features=self.continuous_features,
            categorical_features=self.categorical_features,
        )

        self.assertEqual(len(variables), len(widgets) - 1)
        self.assertTrue(all([isinstance(v, str) for v in variables]))
        self.assertTrue(all([isinstance(v, dict) for v in variables.values()]))

        self.assertTrue(all(["id" in v.keys() for v in variables.values()]))
        self.assertTrue(all(["caption" in v.keys() for v in variables.values()]))
        self.assertTrue(all(["type" in v.keys() for v in variables.values()]))
        self.assertTrue(
            all(
                [
                    "options" in v.keys()
                    for v in variables.values()
                    if v["type"] == "categorical"
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    "value" in v.keys()
                    for v in variables.values()
                    if v["type"] == "continuous"
                ]
            )
        )

    def test_update_values(self):
        pass


if __name__ == "__main__":
    unittest.main()
