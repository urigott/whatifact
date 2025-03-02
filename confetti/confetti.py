"""
Main confetti function and Shiny code
"""

from pathlib import Path

import pandas as pd
import numpy as np
from shiny import App, Inputs, Outputs, Session, render, ui, reactive, run_app, session

from confetti.widgets import _set_null_in_variable, _un_null_variable
from confetti.preprocessing import _get_variables_and_widgets, _update_values
from confetti.inference import _prepare_everything

CSS_FILE = Path(__file__).parent / "css" / "style.css"


def confetti(
    df,
    clf,
    sample_id: str = None,
    feature_settings: dict = None,
    run_application: bool = True,
):  # pylint: disable=too-many-arguments
    """
    Creates a UI for counterfactual explorations, allowing users to modify input features
    and observe changes in model predictions.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing the input features.
    clf : object
        A trained classification model that supports prediction and has a predict_proba method.
    sample_id : str, optional
        The identifier for a specific sample in the dataset (default is None).
    feature_settings : dict, optional
        A dictionary specifying constraints or custom settings for feature modifications.
        Will be inferenced if None.
    run_application : bool, optional
        Whether to launch the UI application immediately (default is True), or return an App object.

    """

    # assert data, prepare date
    (df, sample_id_type, continuous_features, categorical_features) = (
        _prepare_everything(
            df=df,
            clf=clf,
            sample_id=sample_id,
            feature_settings=feature_settings,
        )
    )

    # build widgets
    variables, widgets = _get_variables_and_widgets(
        df=df,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        feature_settings=feature_settings,
    )

    ## SHINY APP
    app_ui = ui.page_auto(
        ui.include_css(CSS_FILE),
        ui.card(
            ui.div(
                ui.row(
                    ui.column(9, ui.output_text_verbatim("calc_pred")),
                    ui.column(
                        2,
                        ui.input_action_button(
                            id="exit", label="X", class_="exit-button"
                        ),
                        title="Close ConFETti",
                    ),
                ),
                ui.row(
                    ui.input_select(
                        id="sample_id",
                        label="Sample ID",
                        choices=list(df.index.astype(str)),
                    )
                ),
                class_="fixed-header",
            ),
            ui.div(*widgets.values(), class_="scrollable-card"),
        ),
        title="ConFETti",
    )

    def server(
        inputs: Inputs, output: Outputs, session: Session
    ):  # pylint: disable=unused-argument
        null_checkboxes = [
            n
            for n in inputs._map.keys()
            if n.startswith("var")
            and n.endswith("_null")  # pylint: disable=protected-access
        ]

        def get_null_checkboxes_dict():
            with reactive.isolate():
                return {n[:-5]: inputs[n]() for n in null_checkboxes}

        def get_params():
            null_checkboxes_dict = get_null_checkboxes_dict()
            params = pd.DataFrame(
                {
                    v["caption"]: (
                        [inputs[var_id]()]
                        if not null_checkboxes_dict.get(var_id, False)
                        else [np.nan]
                    )
                    for var_id, v in variables.items()
                }
            )

            # Handle categorical features
            for col in categorical_features:
                if (df[col].dtype == "category") and (params.loc[0, col] != ""):
                    cat_dtype = df[col].cat.categories.dtype
                    params[col] = params[col].astype(cat_dtype)

            params = params.astype(df.dtypes.replace({int: "float64"}))

            return params

        def get_active_sample_id():
            with reactive.isolate():
                return sample_id_type(inputs.sample_id())

        state = reactive.Value()
        state.dict = get_null_checkboxes_dict()

        @reactive.effect
        def checkbox_change():
            null_checkboxes_dict = {n[:-5]: inputs[n]() for n in null_checkboxes}

            for var_id, null_value in null_checkboxes_dict.items():

                if (
                    null_value == state.dict[var_id]
                ):  # if checkbox is unchanged, continue
                    continue

                if null_value:
                    _set_null_in_variable(var_id)
                else:
                    _un_null_variable(
                        var_id,
                        var_dict=variables[var_id],
                        org_value=df.loc[
                            get_active_sample_id(), variables[var_id]["caption"]
                        ],
                    )

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

        @reactive.calc
        @render.text()
        def calc_pred() -> str:
            params = get_params()

            # # ONLY IN DEBUGGING MDOE
            # print(f'PARAMS: {params.round(1).to_dict(orient='records')}')

            try:
                prediction = clf.predict_proba(params)[:, 1].item()

            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"predict_proba failed with error: {str(e)}")
                exit_app()

            prediction_string = f"PREDICTION PROBABILITY: {prediction:.3f}"
            return prediction_string

        @reactive.effect
        @reactive.event(inputs.exit)
        async def exit_app():
            await session.close()

    app = App(app_ui, server)

    if run_application:
        run_app(app, launch_browser=True)

    return app
