import unittest

import numpy as np

from whatifact._widgets import _get_single_slider, _get_drop_down, _get_null_checkbox


class TestContinuous(unittest.TestCase):
    def setUp(self):
        self.arr1 = np.linspace(-5, 5, 100)
        self.arr2 = np.arange(10)

        mask = np.random.binomial(p=0.1, n=1, size=self.arr1.shape)
        self.arr3 = self.arr1.copy()
        self.arr3[mask] = np.nan

        self.v1 = dict(
            id="id", caption="caption", min=1, max=10, value=3, step=1, null=True
        )
        self.v2 = dict(
            id="id", caption="caption", min=1, max=10, value=3, step=1, null=False
        )

    def test_get_single_slider_allow_nulls(self):
        slider = _get_single_slider(v=self.v1)
        html = slider.get_html_string()

        self.assertIn(f'data-min="{self.v1["min"]}"', html)
        self.assertIn(f'data-max="{self.v1["max"]}"', html)
        self.assertIn(f"checkbox", html)

    def test_get_single_slider_allow_nulls_false(self):
        slider = _get_single_slider(v=self.v2)
        html = slider.get_html_string()

        self.assertIn(f'data-min="{self.v2["min"]}"', html)
        self.assertIn(f'data-max="{self.v2["max"]}"', html)
        self.assertNotIn(f"checkbox", html)

    def test_get_null_checkbox(self):
        checkbox = _get_null_checkbox(v=self.v1)
        html = checkbox.get_html_string()

        self.assertIn("id_null", html)


class TestCategorical(unittest.TestCase):
    def setUp(self):
        self.v = dict(
            id="id", caption="caption", options=["a", "b", "c"], value="b", null=False
        )

    def test_get_drop_down(self):
        selector = _get_drop_down(self.v)
        html = selector.get_html_string()
        for option in self.v["options"]:
            self.assertIn(f'value="{option}"', html)


if __name__ == "__main__":
    unittest.main()
