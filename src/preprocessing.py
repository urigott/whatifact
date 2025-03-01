"""
Data preprocessing functions
"""

from typing import List, Dict, Union

from shiny import ui
import pandas as pd

from src.widgets import (
    _get_drop_down,
    _get_single_slider,
    _set_null_in_variable,
    _un_null_variable,
)
from src.inference import _get_sliders_params, _get_select_list_params


def _get_variables_and_widgets(
    df,
    continuous_features: List[str],
    categorical_features: List[str],
    feature_settings: Dict[str, Dict[str, Union[int, float]]] = None,
):
    variables = dict() # pylint: disable=use-dict-literal
    widgets = dict() # pylint: disable=use-dict-literal
    feature_settings = feature_settings or dict() # pylint: disable=use-dict-literal

    for j, col in enumerate(df.columns, start=1):
        var_id = f"var{j}"
        var_dict = {"id": var_id, "caption": col}

        if col in continuous_features:
            var_dict.update(
                **_get_sliders_params(df[col], col_dict=feature_settings.get(col, dict())) # pylint: disable=use-dict-literal
            )

        elif col in categorical_features:
            var_dict.update(
                **_get_select_list_params(
                    df[col], col_dict=feature_settings.get(col, dict()) # pylint: disable=use-dict-literal
                )
            )

        variables.update({var_id: var_dict})
        widgets.update(
            {
                var_id: (
                    _get_single_slider(v=var_dict)
                    if var_dict["type"] == "continuous"
                    else _get_drop_down(v=var_dict)
                )
            }
        )

    widgets.update({"footer": ui.card_footer(" ")})

    return variables, widgets


def _update_values(df, sample, variables):
    row = df.loc[[sample]]
    for var_id, v in variables.items():
        col = v["caption"]
        new_value = row.loc[sample, col]

        if v["type"] == "continuous":
            if pd.isna(new_value):
                _set_null_in_variable(var_id)
                ui.update_checkbox(var_id + "_null", value=True)
            else:
                _un_null_variable(var_id, variables[var_id], org_value=new_value)
                ui.update_slider(var_id, value=new_value, min=v["min"], max=v["max"])
                ui.update_checkbox(var_id + "_null", value=False)

        if v["type"] == "categorical":
            ui.update_select(
                id=v["id"], selected=str(new_value) if not pd.isna(new_value) else ""
            )
