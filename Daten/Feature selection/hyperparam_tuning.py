import numpy as np
import pandas
from sklearn_genetic import GAFeatureSelectionCV

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

csv_path = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Datensätze\Trab_data.csv'
df = pandas.read_csv(csv_path, sep=';')
df = df.drop('dog_id', axis=1)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = DecisionTreeClassifier()
cv = StratifiedKFold(shuffle=True)

# hyperparams
n_gens = 30
mutation_p = 0.01

best_acc = -1
best_params = None

while n_gens <= 100:
    while mutation_p <= 0.1:
        ea_estm = GAFeatureSelectionCV(
            estimator=clf,
            cv=cv,
            scoring="f1",
            generations=n_gens,
            population_size=50,
            crossover_probability=0.5,
            mutation_probability=mutation_p,
            verbose=True,
            elitism=True,
            keep_top_k=2,
            n_jobs=-1,
        )
        print('mutation:', mutation_p, 'generations:', n_gens)
        ea_estm.fit(X, y)
        if np.max(ea_estm.history['fitness_max']) >= best_acc:
            best_acc = np.max(ea_estm.history['fitness_max'])
            best_params = (n_gens, mutation_p)
        mutation_p += 0.01
    n_gens += 10
    mutation_p = 0.01

print('\nbest accuracy: ', best_acc, '\nbest params: ', best_params)
