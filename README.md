# whatifact
## A Counter Factual Exploration Tool

Counter Factual explanations are a methdology to investigate model's predictions.
whatifact allows data scientists to play with their population features, and find out how predictions are changed.

Specifically, it allows to ask "what-if" questions: 
- What would the model predict if a record belonged to a man, instead of a woman?
- What would the model predict if my blood glucose levels were slightly higher, or lower?

It should be noted that this tool can help assess the causal questions of the *model's prediction* - but not the causal questions of the real world!
Answer causal questions of the real worlds require unique design, rather than some UI tool...

## Example

In the most basic setting, whatifact only requries the data and a classifier.
Everything will be selected automatically: 
- Whether a feature is categorical or continuous
- Should missing values be allowed
- How to set-up the sliders for continuous features

You will notice that some sliders have a little checkbox on their left. 
Selecting this checkbox will disable the slider, and set the value for this feature as null.

```{python}
from sklearn.datasets import fetch_openml
import lightgbm as lgb

from whatifact import whatifact

# Load Titanic dataset
titanic = fetch_openml("titanic", version=1, as_frame=True)

# Convert to DataFrame
df = titanic.data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
labels = titanic.target.astype(int)

# Train a LGBMClassifier
clf = lgb.LGBMClassifier(verbose=0).fit(df, labels)

# Running whatifact
app = whatifact(df=df, clf=clf)

# # Output: (Clicking on the http link will open whatifact in the browser)
# INFO:     Started server process [42841]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://<LOCAL_IP>:8000 (Press CTRL+C to quit)
```

However, when running the code above, you will notice a strange behavior.
The sliders for both `age` and `fare` start at negative values, which is of cource non-sensical, the `parch` variable has no missing-value checkbox next to it, and the `sibsp` was considered as a continuous feature, but we'd rather handle it as categorical.

To change this behavior, we can send the `feature_settings` parameter, such as:

```{python}
feature_settings = {
    'parch': {'null': True},
    'age': {'min': 0},
    'fare': {'min': 0},
    'sibsp': {'type': 'categorical'}
}

app = whatifact(df=df, clf=clf, feature_settings=feature_settings)
```

This should solve the above behavior.
`feature_settings` is a dictionary, with column names as keys and dicionaries as values. 
All features may contain `null` or `type` keys.
* `null`: a boolearn (True/False) to state whether the feature be null (i.e., null-checkbox for continuous features, an empty selection for categorical features).
* `type`: 'continuous' or 'categorical' - manually defining the feature type.

Continuous features may also contain the `min`, `max`, `step`, and `decimals` keys. All other keys will be ignored.
The `decimals` parameters is an integer defining the number of decimal digits for rounding purposes (default: 1).

The last two parameters in whatifact are `sample_id` and `run_application`.
* `sample_id` is the name of a column in `df`, that will be used in the sample selector at the top of the app. If it remains None, the index column will be used as sample_id.
* `run_application` is a boolean (True/False, defaults to True) that run the web service to run the shiny app. If changed to False, an App object will be returned, but not run, and running the app will require `shiny run my_file.py`

## Limitations
whatifact currently works with binary prediction models only, and should support LogisticRegression, XGBoost, and LGBMClassifier.



