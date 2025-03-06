"""
Shiny widgets are defined here
"""

import numpy as np
import pandas as pd
from shiny import ui

def _get_null_checkbox(v, value=False):
    return ui.column(1, ui.input_checkbox(id=v["id"] + "_null", label="", value=value))

def _get_single_slider(v: dict):
    row = ui.row()

    if v["null"]:
        initial_value = ~np.isnan(v["value"])
        row.append(_get_null_checkbox(v, value=initial_value))
    else:
        row.append(ui.column(1, ""))

    row.append(
        ui.column(
            9,
            ui.input_slider(
                id=v["id"],
                label=v["caption"],
                min=v["min"],
                max=v["max"],
                value=float(v["value"]),
                step=v["step"],
            ),
        )
    )
    return row


def _get_drop_down(v):
    row = ui.row()

    if v["null"]:
        initial_value = v["value"] != ""
        row.append(_get_null_checkbox(v, value=initial_value))
    else:
        row.append(ui.column(1, ""))

    row.append(
        ui.column(
            9,
            ui.input_select(
                v["id"], v["caption"], choices=v["options"], selected=v["value"]
            ),
        )
    )

    return row


def _disable_slider(var_id):
    ui.update_slider(var_id, min=0, max=0)


def _enable_slider(var_id, var_dict, org_value):
    value = max(var_dict["min"], org_value)
    ui.update_slider(
        var_id, min=var_dict["min"], max=var_dict["max"], value=float(value)
    )


def _get_card_header(df: pd.DataFrame, sample_id: str = None):
    card_header = ui.card_header(
        ui.div(
            ui.input_select(
                id="sample_id",
                label=sample_id or "Sample ID",
                choices=list(df.index.astype(str)),
            ),
            class_="center-select",
        ),
        ui.div(
            "Predicted probability",
            ui.row(ui.output_text_verbatim("calc_pred")),
            class_="prediction-box",
        ),
        # ui.input_action_button(id='revert', label='⟳')
        class_="fixed-header",
    )
    return card_header
