"""
Main confetti function and Shiny code
"""

from pathlib import Path

import pandas as pd
import numpy as np
from shiny import App, Inputs, Outputs, Session, render, ui, reactive, run_app, session

from confetti.widgets import _disable_slider, _enable_slider
from confetti.preprocessing import _get_variables_and_widgets, _update_values
from confetti.inference import _prepare_everything

from confetti.logger import get_logger

CSS_FILE = Path(__file__).parent / "resources" / "style.css"


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
    app_ui = ui.page_fluid(
        ui.include_css(CSS_FILE),
        ui.card(
            ui.card_header(
                ui.div(
                    ui.input_select(
                        id="sample_id",
                        label=sample_id or "Sample ID",
                        choices=list(df.index.astype(str))
                    ), 
                    class_='center-select'
                ),
                ui.div("Predicted probability",                    
                    ui.row(ui.output_text_verbatim("calc_pred")),
                    class_="prediction-box",
                ),
                # ui.input_action_button(id='revert', label='⟳')
                
                class_="fixed-header",
            ),
            ui.card_body(ui.div(*widgets.values(), class_="scrollable-card")),
        ),
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
                null_dict = {n[:-5]: inputs[n]() for n in null_checkboxes}
                logger.info(f"CALLED get_null_checkboxes_dict | {str(null_dict)}")
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
            logger.info("CALLED get_params | " + str(params.to_dict(orient="records")))
            return params

        def get_active_sample_id():
            with reactive.isolate():
                new_sample_id = sample_id_type(inputs.sample_id())
                logger.info(f"SAMPLE_ID: {new_sample_id}")
                return new_sample_id

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
            logger.info(f"CALLED checkbox_change: {state.dict}")

        @reactive.effect        
        def sync_sample_id():
            logger.info("CALLED {sync_sample_id}")
            sample = sample_id_type(inputs["sample_id"]())
            _update_values(
                df=df,
                sample=sample,
                variables=variables,
            )

        @reactive.effect
        @reactive.event(inputs.revert)
        def revert_changes():
            logger.info("CALLED {revert_changes}")
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

            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(f"ERROR calculating predictions: {str(e)}")
                exit_app()

            prediction = np.round(prediction, 3)
            logger.info(f"CALLED calc_pred {prediction:.3f}")
            return prediction

        @reactive.effect
        @reactive.event(inputs.exit)
        async def exit_app():
            logger.info("EXIT")
            await session.close()

    app = App(app_ui, server)
    logger = get_logger()

    if run_application:
        run_app(app, launch_browser=True)

    return app
