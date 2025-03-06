from sklearn.datasets import fetch_openml
import lightgbm as lgb

from antifact.antifact import antifact

# Load Titanic dataset
titanic = fetch_openml("titanic", version=1, as_frame=True)

# Convert to DataFrame
df = titanic.data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
labels = titanic.target.astype(int)

# Train a LGBMClassifier
clf = lgb.LGBMClassifier(verbose=0).fit(df, labels)

feature_settings = {
    'parch': {'null': True},
    'age': {'min': 0},
    'fare': {'min': 0},
    'sibsp': {'type': 'categorical', 'null': False}

}

# Running antifact
app = antifact(df=df, clf=clf, feature_settings=feature_settings)