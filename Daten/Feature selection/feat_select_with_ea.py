from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn_genetic import GAFeatureSelectionCV, ExponentialAdapter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

csv_path = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Datensätze\Trab_data.csv'
raw_df = pd.read_csv(csv_path, sep=';')
raw_df.drop(columns=['dog_id', 'level'], inplace=True)
for i in range(1, 6):
    raw_df[f'random_{i}'] = np.random.rand(raw_df.shape[0]) * 10

scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(raw_df), columns=raw_df.columns)

X = df.drop(columns='target')
y = raw_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtc = DecisionTreeClassifier()
svc = SVC()
knn = KNeighborsClassifier()

cv = StratifiedKFold(shuffle=True, random_state=42)

mutation_scheduler = ExponentialAdapter(0.05, 0.001, 0.001)
crossover_scheduler = ExponentialAdapter(0.2, 0.9, 0.05)

ea_estm = GAFeatureSelectionCV(
    estimator=svc,
    cv=cv,
    scoring="accuracy",
    generations=50,  # 70,
    population_size=100,
    crossover_probability=crossover_scheduler,  # 0.7,
    mutation_probability=mutation_scheduler,  # 0.05,
    verbose=True,
    elitism=True,
    keep_top_k=2,
    n_jobs=-1,
    tournament_size=5
)

ea_estm.fit(X_train.values, y_train.values)
print(len(ea_estm.n_features), 'total features seen')
print('features:', ea_estm.support_)
sel_feats = ea_estm.get_feature_names_out(list(X))
print('best_feat:', sel_feats, '(', len(sel_feats), ')')

print(ea_estm.best_features_)
# print(ea_estm.get_params())
# -------training done------- #

# X_test = ea_estm.transform(X_test)
y_pred = ea_estm.predict(X_test.values)

print('acc:', accuracy_score(y_test, y_pred))
