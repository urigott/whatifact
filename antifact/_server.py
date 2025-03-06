from shiny import ui, reactive, Inputs, Outputs, Session, render
import numpy as np
import pandas as pd

from antifact._utilities import _disable_slider, _enable_slider
from antifact._utilities import _update_values


def antifact_server(
        inputs: Inputs, 
        output: Outputs, 
        session: Session, 
        variables: dict, 
        sample_id_type, 
        df: pd.DataFrame, 
        clf: object
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
                # logger.info(f"CALLED get_null_checkboxes_dict | {str(null_dict)}")
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
            # logger.info("CALLED get_params | " + str(params.to_dict(orient="records")))
            return params

        def get_active_sample_id():
            with reactive.isolate():
                new_sample_id = sample_id_type(inputs.sample_id())
                # logger.info(f"SAMPLE_ID: {new_sample_id}")
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
            # logger.info(f"CALLED checkbox_change: {state.dict}")

        @reactive.effect        
        def sync_sample_id():
            # logger.info("CALLED {sync_sample_id}")
            sample = sample_id_type(inputs["sample_id"]())
            _update_values(
                df=df,
                sample=sample,
                variables=variables,
            )

        @reactive.effect
        @reactive.event(inputs.revert)
        def revert_changes():
            # logger.info("CALLED {revert_changes}")
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
                # logger.warning(f"ERROR calculating predictions: {str(e)}")
                exit_app()

            prediction = np.round(prediction, 3)
            # logger.info(f"CALLED calc_pred {prediction:.3f}")
            return prediction

        @reactive.effect
        @reactive.event(inputs.exit)
        async def exit_app():
            # logger.info("EXIT")
            await session.close()