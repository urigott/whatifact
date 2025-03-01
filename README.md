# ConFETti
## A Counter Factual Exploration Tool

Counter Factual explanations are a methdology to investigate model's predictions.
ConFETti allows data scientists to play with their population features, and find out how predictions are changed.

Specifically, it allows to ask "what-if" questions: 
- What would the model predict if a record belonged to a man, instead of a woman?
- What would the model predict if my blood glucose levels were slightly higher, or lower?

It should be noted that this tool can help assess the causal questions of the *model's prediction* - but not the causal questions of the real world!
Answer causal questions of the real worlds require unique design, rather than some UI tool...

## Limitations
ConFETti currently works with binary prediction models only, and should support LogisticRegression, XGBoost, and LGBMClassifier.

## Example

[TODO]

## Known issues and future plans
- Display SHAP values for predictions
- Allow multi-class classifiers and regression models




