"""
Shiny widgets are defined here
"""

import numpy as np
from shiny import ui

from confetti.logger import get_logger

logger = get_logger()

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
    logger.info(f'CALLED _get_drop_down | {str(v)}')
    row = ui.row()    

    if v["null"]:
        initial_value = v["value"] != ""
        row.append(_get_null_checkbox(v, value=initial_value))
    else:
        row.append(ui.column(1, ""))
        

    row.append(ui.column(9, ui.input_select(
        v["id"], v["caption"], choices=v["options"], selected=v["value"]
    )))

    return row


def _disable_slider(var_id):
    ui.update_slider(var_id, min=0, max=0)


def _enable_slider(var_id, var_dict, org_value):
    value = max(var_dict["min"], org_value)
    ui.update_slider(
        var_id, min=var_dict["min"], max=var_dict["max"], value=float(value)
    )
