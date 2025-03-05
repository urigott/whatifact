from sklearn.datasets import fetch_openml
import lightgbm as lgb

import sys
sys.path.insert(0, '/home/urigott/confetti/')

from confetti.confetti import confetti

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

# Running confetti
app = confetti(df=df, clf=clf, feature_settings=feature_settings)