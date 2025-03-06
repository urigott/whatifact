"""
Main whatifact function and Shiny code
"""

from pathlib import Path

from shiny import App, Inputs, Outputs, Session, ui, run_app

from whatifact._widgets import _get_card_header
from whatifact._utilities import _get_variables_and_widgets
from whatifact._inference import _prepare_everything
from whatifact._server import whatifact_server

CSS_FILE = Path(__file__).parent / "resources" / "style.css"


def whatifact(
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

    # Build UI
    app_ui = ui.page_fluid(
        ui.include_css(CSS_FILE),
        ui.card(
            _get_card_header(df, sample_id),
            ui.card_body(ui.div(*widgets.values(), class_="scrollable-card")),
        ),
    )

    def app_server(inputs: Inputs, outputs: Outputs, session: Session):
        whatifact_server(inputs, outputs, session, variables, sample_id_type, df, clf)

    app = App(app_ui, app_server)

    if run_application:
        run_app(app, launch_browser=True)

    return app
