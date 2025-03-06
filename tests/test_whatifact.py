import unittest
import warnings

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from whatifact import whatifact


class TestWhatifact(unittest.TestCase):
    def setUp(self):
        n = 500
        self.df = pd.DataFrame(
            {
                "sample_id": np.random.choice(10000, size=n, replace=False),
                "Age": pd.Series(np.random.randint(25, 80, size=n)),
                "Height": pd.Series(np.random.randint(159, 191, size=n)),
                "Sex": pd.Series(
                    np.random.choice(["M", "F"], size=n, replace=True)
                ).astype("category"),
                "Hb": pd.Series(np.random.normal(loc=12, size=n)),
                "Diabetic": pd.Series(np.random.binomial(n=2, p=0.25, size=n)).astype(
                    "category"
                ),
            }
        )

        # preparing data for Logisitic regression
        self.df_numerical = self.df.copy()
        self.df_numerical["Sex"] = (
            self.df_numerical["Sex"].map({"M": 0, "F": 1}).astype(int)
        )
        self.df_numerical["Diabetic"] = self.df_numerical["Diabetic"].astype(int)

        # preparing outcomes
        outcome = (
            self.df["Age"] / self.df["Height"]
            + (self.df["Hb"] * self.df["Diabetic"].astype(int))
        ) * self.df["Sex"].apply(lambda x: 1 if x == "M" else 1.5).astype(int)
        prediction_threshold = np.quantile(outcome, q=0.75)
        self.labels = (outcome > prediction_threshold).astype(int)

    def test_with_lgbm(self):
        clf = LGBMClassifier(verbose=-1).fit(
            self.df.drop("sample_id", axis=1), self.labels
        )
        _ = whatifact(
            df=self.df,
            clf=clf,
            sample_id="sample_id",
            run_application=False,
        )

    def test_with_lgbm_with_missing_values(self):
        mask = np.random.binomial(n=1, p=0.05, size=self.df.shape).astype(bool)
        df_array = self.df.values
        df_array[mask] = np.nan
        df_missing = pd.DataFrame(df_array, columns=self.df.columns)
        df_missing["sample_id"] = self.df[
            "sample_id"
        ]  # sample_id should not have missing values
        df_missing = df_missing.astype(
            {
                "Age": float,
                "Height": float,
                "Sex": "category",
                "Hb": float,
                "Diabetic": "category",
            }
        )

        self.assertGreater(
            df_missing.isna().sum().sum(), 0
        )  # assert df_missing really has missing values

        clf = LGBMClassifier(verbose=-1).fit(
            df_missing.drop("sample_id", axis=1), self.labels
        )
        _ = whatifact(
            df=df_missing,
            clf=clf,
            sample_id="sample_id",
            run_application=False,
        )

    def test_with_logistic_regression(self):
        warnings.filterwarnings("ignore")
        clf = LogisticRegression(verbose=0).fit(
            self.df_numerical.drop("sample_id", axis=1), self.labels
        )
        _ = whatifact(
            df=self.df_numerical,
            clf=clf,
            sample_id="sample_id",
            run_application=False,
        )


if __name__ == "__main__":
    unittest.main()
