import numpy as np
import pandas
from sklearn.datasets import load_iris
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

csv_path = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Datensätze\Trab_data.csv'
df = pandas.read_csv(csv_path, sep=';')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
clf = DecisionTreeClassifier()
cv = StratifiedKFold(shuffle=True)
ea_estm = GAFeatureSelectionCV(
    estimator=clf,
    cv=cv,
    scoring="accuracy",
    generations=50,
    mutation_probability=0.7,
    verbose=True,
    elitism=True,
    keep_top_k=2
)

ea_estm.fit(X, y)
print(ea_estm.best_features_)

print('end')
