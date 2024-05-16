import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform, uniform
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, \
    RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    csv_path = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\feature_df.csv'  # ready and processed data set
    data = pd.read_csv(csv_path, sep=',')
    data.drop('dog_id', axis=1, inplace=True)

    # scale all except target
    df_no_target = data.drop('target', axis=1).columns
    data[df_no_target] = StandardScaler().fit_transform(data[df_no_target])

    # split into train-test set
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

    ###############################
    ###############################

    # classifiers
    lr = LogisticRegression(max_iter=1000)
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    penalty = ['none', 'l1', 'l2', 'elasticnet']
    l1_ratio = [i / 100 for i in range(1, 101)]
    # c_values = [10000, 1000, 100, 10, 1.0, 0.1, 0.01, 0.001]
    c_values = loguniform(1e-5, 1000)  # for random search
    grid = dict(solver=solvers, penalty=penalty, C=c_values, l1_ratio=l1_ratio)
    ##########
    knn = KNeighborsClassifier()
    metric = ['euclidean', 'manhattan', 'minkowski']
    weights = ['uniform', 'distance']
    p = [1, 2]
    n_neighbors = range(1, 100, 2)
    # grid = dict(metric=metric, n_neighbors=n_neighbors, weights=weights, p=p)
    ##########
    svc = SVC(cache_size=1200)
    # c = [1000, 100, 10, 1.0, 0.1, 0.01, 0.001]
    c = loguniform(1e-5, 1000)
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma = ['scale', 'auto']
    degree = range(1, 11)
    # grid = dict(kernel=kernel, C=c, degree=degree)
    ###############################
    ###############################

    # hyper opt
    model = lr
    # default training as baseline
    baseline = False
    if baseline:
        default = model
        print('default', default)
        default.fit(X_train, y_train)

        y_pred = default.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        f1 = f1_score(y_pred, y_test)
        print("\tAccuracy:", acc)
        print("\tF1 Score:", f1)

    rnd = True
    if not rnd:  # grid
        print('Grid')
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1',
                                   error_score=0)
        grid_result = grid_search.fit(X_train, y_train)
        res = grid_result
    else:  # random
        print('\nRandom')
        rnd_search = RandomizedSearchCV(model, grid, n_iter=1000, scoring='f1', n_jobs=-1, cv=cv,
                                        random_state=42)
        random_result = rnd_search.fit(X_train, y_train)
        res = random_result

    # evaluation
    print(res.score)
    means = res.cv_results_['mean_test_score']
    stds = res.cv_results_['std_test_score']
    params = res.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        if mean:  # exclude nan
            print("%f (%f) with: %r" % (mean, stdev, param))

    print(model)
    print("\nBest: %f using %s" % (res.best_score_, res.best_params_))

    model.set_params(**res.best_params_)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test)
    print("Accuracy:", acc)
    print("F1 Score:", f1)

    print('----------hyperparams. tuned----------')

    return res.best_params_


if __name__ == '__main__':
    main()
