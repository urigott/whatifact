from shiny import App, reactive, render, ui, run_app

app_ui = ui.page_fluid(
    ui.input_checkbox("checkbox_1", "Checkbox 1", False),
    ui.input_checkbox("checkbox_2", "Checkbox 2", False),
    ui.input_checkbox("checkbox_3", "Checkbox 3", False),
    ui.output_text_verbatim("output"),
)

def server(input, output, session):
    # List of checkbox IDs
    checkbox_ids = ["checkbox_1", "checkbox_2", "checkbox_3"]

    # This effect will run whenever any of the checkboxes changes
    @reactive.effect
    def _():
        # Check for changes in any checkbox
        for checkbox_id in checkbox_ids:
            if input[checkbox_id]() != input[checkbox_id]():  # Check if it has changed
                print(f"{checkbox_id} changed!")

    # Output to display the current checkbox statuses
    @render.text
    def output():
        status = [f"{checkbox_id}: {input[checkbox_id]()}" for checkbox_id in checkbox_ids]
        return "\n".join(status)

app = App(app_ui, server)
run_app(app, launch_browser=True)