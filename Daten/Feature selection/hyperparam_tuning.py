import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform, uniform
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, \
    RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    csv_path = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\feature_df.csv'  # ready and processed data set
    data = pd.read_csv(csv_path, sep=',')
    data.drop('dog_id', axis=1, inplace=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='hpo.log',
                        filemode='w')

    # scale all except target
    df_no_target = data.drop('target', axis=1).columns
    data[df_no_target] = StandardScaler().fit_transform(data[df_no_target])

    # split into train-test set
    X = data.drop('target', axis=1)
    y = data['target']
    # make 70/20/10 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    ###############################
    ###############################

    # classifiers
    lr = LogisticRegression(max_iter=1000)
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']  # 5
    penalty = ['none', 'l1', 'l2', 'elasticnet']  # 4
    l1_ratio = [i / 100 for i in range(1, 101)]  # 100
    # c_values = [10000, 1000, 100, 10, 1.0, 0.1, 0.01, 0.001]
    c_values = loguniform(1e-5, 10000)  # for random search
    lr_grid = dict(solver=solvers, penalty=penalty, C=c_values, l1_ratio=l1_ratio)
    ##########
    knn = KNeighborsClassifier()
    metric = ['euclidean', 'manhattan', 'minkowski']  # 3
    weights = ['uniform', 'distance']  # 2
    p = uniform(1, 2)  # [1, 2]  # 2
    n_neighbors = range(1, 101)  # 100
    knn_grid = dict(metric=metric, n_neighbors=n_neighbors, weights=weights, p=p)
    ##########
    svc = SVC(cache_size=1200)
    # c = [1000, 100, 10, 1.0, 0.1, 0.01, 0.001]
    c = loguniform(1e-5, 10000)
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']  # 4
    gamma = loguniform(1e-5, 10000)  # ['scale', 'auto']
    degree = range(1, 11)  # 10
    svc_grid = dict(kernel=kernel, C=c, degree=degree, gamma=gamma)
    ###############################
    ###############################

    # hyper opt
    models = [lr, knn, svc]
    grids = [lr_grid, knn_grid, svc_grid]
    # model = knn
    # default training as baseline
    default_results = defaultdict(dict)
    baseline = False
    if baseline:
        for model in models:
            print('default', model)
            scores_acc = cross_val_score(model, X_val, y_val, scoring='accuracy', cv=cv, n_jobs=-1)
            scores_f1 = cross_val_score(model, X_val, y_val, scoring='f1', cv=cv, n_jobs=-1)

            default_results[model]['acc'] = scores_acc
            default_results[model]['f1'] = scores_f1
            # print("\tAccuracies:", scores_acc)
            print("\tmean Accuracy:", np.mean(scores_acc))
            # print("\tF1 Scores:", scores_f1)
            print("\tmean F1 Score:", np.mean(scores_f1))

    # random or grid search
    tuned_results = []
    for model, grid in zip(models, grids):
        print(model)
        logging.info(model)
        rnd_search = RandomizedSearchCV(model, grid, n_iter=2000, scoring='f1', n_jobs=-1, cv=cv,
                                        random_state=42)
        res = rnd_search.fit(X_val, y_val)
        tuned_results.append(res)
        print("\tBest: %f using %s" % (res.best_score_, res.best_params_))
        logging.info("\tBest: %f using %s" % (res.best_score_, res.best_params_))

    # evaluation
    # eval_print = False
    # if eval_print:
    #     print(res.score)
    #     means = res.cv_results_['mean_test_score']
    #     stds = res.cv_results_['std_test_score']
    #     params = res.cv_results_['params']
    #     for mean, stdev, param in zip(means, stds, params):
    #         if mean:  # exclude nan
    #             print("%f (%f) with: %r" % (mean, stdev, param))

    # hyperparams. tuned
    # lr_hp = {'C': 0.847035189263436, 'penalty': 'l1', 'solver': 'liblinear'}
    # knn_hp = {'weights': 'uniform', 'p': 2, 'n_neighbors': 1, 'metric': 'manhattan'}
    # svc_hp = {'C': 38.83569559468877, 'degree': 6, 'kernel': 'rbf'}
    # print('\nCV on training for tuned:\n')
    # lr_tuned = LogisticRegression().set_params(**lr_hp)
    # knn_tuned = KNeighborsClassifier().set_params(**knn_hp)
    # svc_tuned = SVC().set_params(**svc_hp)
    # tuned_models = [lr_tuned, knn_tuned, svc_tuned]
    # tuned_results = {}
    # for tuned_model in tuned_models:
    #     print('tuned', tuned_model)
    #     scores_acc = cross_val_score(tuned_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    #     scores_f1 = cross_val_score(tuned_model, X_train, y_train, scoring='f1', cv=cv, n_jobs=-1)
    #
    #     tuned_results['acc'] = scores_acc
    #     tuned_results['f1'] = scores_f1
    #     # print("\tAccuracies:", scores_acc)
    #     print("\tmean Accuracy:", np.mean(scores_acc))
    #     # print("\tF1 Scores:", scores_f1)
    #     print("\tmean F1 Score:", np.mean(scores_f1))
    #
    # print('\ndefaults:', default_results)


if __name__ == '__main__':
    main()
