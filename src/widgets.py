import numpy as np
import pandas as pd
from shiny import ui

def _get_null_checkbox(v, value=False):
    return ui.column(1, ui.input_checkbox(
        id=v['id'] + '_null',
        label='',
        value=value
        ))


def _get_single_slider(v: dict):
    row = ui.row()    
    initial_empty = np.isnan(v['value'])    

    if v['null']:
        row.append(_get_null_checkbox(v, value=initial_empty))
        col_width = 9
    else:
        col_width = 12
    
    row.append(ui.column(col_width,
                         ui.input_slider(
                             id=v["id"],
                             label=v["caption"],
                             min=v["min"],
                             max=v["max"],
                             value=v["value"],
                             step=v["step"],
                             )))
    return row

def _get_drop_down(v):    
    return ui.input_select(v["id"], 
                           v['caption'], 
                           choices=v["options"],
                           selected=v['value']
                           )


def _set_null_in_variable(var_id):        
    ui.update_slider(var_id, min=0, max=0)
    
def _un_null_variable(var_id, var_dict, org_value):    
    value = max(var_dict['min'], org_value)
    ui.update_slider(var_id, min=var_dict['min'], max=var_dict['max'], value=value)