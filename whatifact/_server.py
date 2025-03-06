"""
server function for shiny App
"""

from shiny import ui, reactive, Inputs, Outputs, Session, render
import numpy as np
import pandas as pd

from whatifact._utilities import _disable_slider, _enable_slider
from whatifact._utilities import _update_values


def whatifact_server(
    inputs: Inputs,
    output: Outputs,
    session: Session,
    variables: dict,
    sample_id_type,
    df: pd.DataFrame,
    clf: object,
):
    """
    server function for shiny UI app.
    Includes all UI logic
    """
    null_checkboxes = [
        n
        for n in inputs._map.keys()
        if n.startswith("var")
        and n.endswith("_null")  # pylint: disable=protected-access
    ]

    def get_null_checkboxes_dict():
        with reactive.isolate():
            null_dict = {n[:-5]: inputs[n]() for n in null_checkboxes}
            return null_dict

    def get_params():
        null_checkboxes_dict = get_null_checkboxes_dict()
        params_dict = {}

        for var_id, v in variables.items():
            col = v["caption"]
            params_dict.update(
                {
                    col: (
                        [inputs[var_id]()]
                        if ~null_checkboxes_dict.get(var_id, False)
                        else [np.nan]
                    )
                }
            )

            params = pd.DataFrame(params_dict)

            if v["type"] == "categorical" and inputs[var_id]() == "":
                if null_checkboxes_dict[var_id]:
                    ui.update_checkbox(var_id + "_null", value=False)

                if (df[col].dtype == "category") and (params.loc[0, col] != ""):
                    cat_dtype = df[col].cat.categories.dtype
                    params[col] = params[col].astype(cat_dtype)

        params = params.astype(df.dtypes.replace({int: "float64"}))
        return params

    def get_active_sample_id():
        with reactive.isolate():
            new_sample_id = sample_id_type(inputs.sample_id())
            return new_sample_id

    state = reactive.Value()
    state.dict = get_null_checkboxes_dict()

    @reactive.effect
    def checkbox_change():
        null_checkboxes_dict = {n[:-5]: inputs[n]() for n in null_checkboxes}

        for var_id, null_value in null_checkboxes_dict.items():

            if null_value == state.dict[var_id]:  # if checkbox is unchanged, continue
                continue

            if null_value:
                if variables[var_id]["type"] == "continuous":
                    _enable_slider(
                        var_id,
                        var_dict=variables[var_id],
                        org_value=df.loc[
                            get_active_sample_id(), variables[var_id]["caption"]
                        ],
                    )
                else:
                    ui.update_select(
                        var_id,
                        selected=df.loc[
                            get_active_sample_id(), variables[var_id]["caption"]
                        ],
                    )

            else:
                if variables[var_id]["type"] == "continuous":
                    _disable_slider(var_id)
                else:
                    ui.update_select(var_id, selected="")

        # refresh state dict
        state.dict = get_null_checkboxes_dict()

    @reactive.effect
    def sync_sample_id():
        sample = sample_id_type(inputs["sample_id"]())
        _update_values(
            df=df,
            sample=sample,
            variables=variables,
        )

    @reactive.effect
    @reactive.event(inputs.revert)
    def revert_changes():
        sample = sample_id_type(inputs["sample_id"]())
        _update_values(
            df=df,
            sample=sample,
            variables=variables,
        )

    @reactive.calc
    @render.text()
    def calc_pred() -> str:
        params = get_params()

        try:
            prediction = clf.predict_proba(params)[:, 1].item()
            assert isinstance(prediction, float)

        except Exception:  # pylint: disable=broad-exception-caught
            exit_app()

        prediction = np.round(prediction, 3)
        return prediction

    @reactive.effect
    @reactive.event(inputs.exit)
    async def exit_app():
        await session.close()
