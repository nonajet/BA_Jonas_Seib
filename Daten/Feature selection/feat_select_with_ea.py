import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn_genetic import GAFeatureSelectionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn_genetic.plots import plot_fitness_evolution

import hyperparam_tuning


def prepare_data(feat_df, dog_df):
    mapping_dict = {
        'm': {'sex': 0, 'neutered': 0},
        'mk': {'sex': 0, 'neutered': 1},
        'w': {'sex': 1, 'neutered': 0},
        'wk': {'sex': 1, 'neutered': 1}
    }  # split column to binary columns
    dog_df[['sex', 'neutered']] = dog_df['Geschlecht'].map(mapping_dict).apply(pd.Series)
    dog_df = dog_df.dropna(subset=['Krank', 'Rückenlänge', 'Schulterhöhe'])
    # relevant column names from examination table
    rel_cols = ['T-Nr.',
                'Alter',
                'Gewicht',
                'sex',
                'neutered',
                'Rückenlänge',
                'Schulterhöhe',
                'BCS',
                'Krank']
    dog_df = dog_df[rel_cols]

    # fill missing values and convert to int
    dog_df['BCS'] = dog_df['BCS'].apply(lambda x: pd.to_numeric(str(x)[0]) if pd.notna(x) else x)
    imp = SimpleImputer(missing_values=np.nan, strategy="median")
    dog_df['BCS'] = imp.fit_transform(dog_df[['BCS']])
    nan_rows = dog_df[dog_df.isna().any(axis=1)]  # for debug

    dog_df['T-Nr.'] = dog_df['T-Nr.'].astype(str).str.replace('/', '')

    # merge and drop duplicates
    merged = pd.merge(feat_df, dog_df, left_on='dog_id', right_on='T-Nr.', how='inner')
    col_id = merged.pop('dog_id')
    merged.insert(0, 'dog_id', col_id)
    merged['Krank'] = merged['Krank'].astype(int, errors='raise')
    merged.drop('T-Nr.', axis=1, inplace=True)

    mask = (merged['steps_A'] - merged['steps_C']).abs() > 2
    merged = merged[~mask]

    val_counts = merged['Krank'].value_counts()
    merged.rename(columns={"Krank": "target"}, inplace=True)

    # print(merged.dtypes)
    # merged.drop('dog_id', axis=1, inplace=True)

    return merged


def main():
    # csv_path = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Datensätze\Trab_data.csv'
    csv_path = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\dog_features_data_front_back.csv'
    dog_table = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Ganganalyse Alle_20240115.ods'
    np.random.seed(42)
    raw_feat_df = pd.read_csv(csv_path, sep=',')
    dog_df = pd.read_excel(dog_table, engine="odf")

    data = prepare_data(raw_feat_df, dog_df)  # preprocess data
    # data.to_csv('feature_df.csv', index=False)  # save for later application
    data.drop(
        columns=['dog_id', 'ratio_pres2area_A', 'ratio_pres2area_C', 'std_ratio_pres2area_A', 'std_ratio_pres2area_C',
                 'steps_A', 'steps_C'], inplace=True)

    # add random noise features
    for i in range(1, 51):
        data[f'random_{i}'] = np.random.rand(data.shape[0]) * 10

    rebalance = False
    if rebalance:
        # sample instances of class with fewer examples
        target_classes = data['target'].unique()
        # Determine the class with the fewest samples
        min_samples = data['target'].value_counts().min()
        # Sample an equal number of samples from each class
        balanced_df = pd.concat([data[data['target'] == cls].sample(min_samples) for cls in target_classes])
        # Shuffle the DataFrame to mix up the samples from different classes and discard additional samples
        data = balanced_df.sample(frac=1).reset_index(drop=True)

    # scale all except target
    df_no_target = data.drop('target', axis=1).columns
    data[df_no_target] = StandardScaler().fit_transform(data[df_no_target])

    # split data into train-test
    X = data.drop(columns='target')
    y = data['target']
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.5, random_state=4321)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=4242)

    knn = KNeighborsClassifier()
    svc = SVC()
    svc.set_params(**{'C': 38.83569559468877, 'degree': 6, 'kernel': 'rbf'})
    svc_comp = SVC()
    svc_comp.set_params(**{'C': 38.83569559468877, 'degree': 6, 'kernel': 'rbf'})
    svc_comp.fit(X_train, y_train)
    print('SVC w/o FS')
    print('features seen', svc_comp.n_features_in_)
    print('params: :', svc_comp.get_params())
    y_pred = svc_comp.predict(X_test.values)
    print('f1:', f1_score(y_pred, y_test))

    log_reg = LogisticRegression()
    clf = svc
    print('------------------------------------\n\nSVC with FS')
    print('default params: ', clf.get_params())
    # clf.set_params(**hyperparam_tuning.main())  # calc best hyperparams for classifier
    # clf.set_params(**{'metric': 'manhattan', 'n_neighbors': 1, 'weights': 'uniform', 'n_jobs': -1}) # for knn
    print('tuned params.:', clf.get_params())

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1234)

    ea_estm = GAFeatureSelectionCV(
        estimator=clf,
        cv=cv,
        scoring="f1",
        generations=30,  # 70,
        population_size=100,
        # crossover_probability=0.7,
        mutation_probability=1 / len(data.columns),
        verbose=True,
        elitism=True,
        keep_top_k=2,
        n_jobs=-1,
        tournament_size=5
    )

    print('params: :', clf.get_params())

    crossover_p = np.linspace(0.2, 0.8, 10)
    mutation_p = np.linspace(0.05, 0.3, 10)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=123)
    # grid = dict(crossover_probability=crossover_p, mutation_probability=mutation_p)
    # grid_search = GridSearchCV(estimator=ea_estm, param_grid=grid, scoring='f1', n_jobs=-1, cv=cv)
    # grid_result = grid_search.fit(X_train, y_train)

    # evaluation
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    #
    # print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    ea_estm.fit(X_train.values, y_train.values)
    print(ea_estm.n_features, 'total features seen')
    print('features:', ea_estm.support_)
    sel_feats = ea_estm.get_feature_names_out(list(X))
    print('\nbest_feat:', sel_feats)

    # -------training done------- #

    # todo get relevant features to predict --> refit?
    # X_test = ea_estm.transform(X_test)
    y_pred = ea_estm.predict(X_test.values)

    print('acc:', accuracy_score(y_pred, y_test))
    print('f1:', f1_score(y_pred, y_test))

    plot_fitness_evolution(ea_estm)
    plt.show()


if __name__ == '__main__':
    main()
