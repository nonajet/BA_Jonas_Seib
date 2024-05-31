import logging
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn_genetic import GAFeatureSelectionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
from sklearn_genetic.callbacks import ProgressBar


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


def save_ga_results(fitness_data, nr, filename='fitness_results.csv'):
    # Convert fitness data to a DataFrame
    df = pd.DataFrame(fitness_data)
    df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)


def main():
    csv_path = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\dog_features_data_front_back.csv'
    dog_table = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Ganganalyse Alle_20240115.ods'
    np.random.seed(100)
    raw_feat_df = pd.read_csv(csv_path, sep=',')
    dog_df = pd.read_excel(dog_table, engine="odf")
    exp_name = 'fss4'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s', filename=f'{exp_name}.log',
                        filemode='w')

    data = prepare_data(raw_feat_df, dog_df)  # preprocess data
    # data.to_csv('feature_df.csv', index=False)  # save for later application
    data.drop(
        columns=['dog_id', 'ratio_pres2area_A', 'ratio_pres2area_C', 'std_ratio_pres2area_A', 'std_ratio_pres2area_C',
                 'steps_A', 'steps_C'], inplace=True)

    # add random noise features
    for i in range(1, 11):
        data[f'RANDOM_{i}'] = np.random.rand(data.shape[0]) * 10000

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
    # make 70/20/10 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    # naive classifier
    print('------------------------------------NC------------------------------------\n\n')
    majority_class = y_train.mode()[0]

    # Create a naive classifier that always predicts the majority class
    class NaiveClassifier:
        def fit(self, X, y):
            self.majority_class = y.mode()[0]

        def predict(self, X):
            return np.full(shape=(X.shape[0],), fill_value=self.majority_class)

    # train naive classifier
    naive_clf = NaiveClassifier()
    naive_clf.fit(X_train, y_train)
    # predict on test set
    y_pred = naive_clf.predict(X_test)
    # evaluate naive classifier
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f'Accuracy: {accuracy:.3f}')
    print(f'F1 Score: {f1:.3f}')
    print('------------------------------------NC done------------------------------------\n\n')

    # hyperparams. tuned
    lr_hp = {'C': 2.2819037991344584, 'l1_ratio': 0.93, 'penalty': 'elasticnet',
             'solver': 'saga'}  # {'C': 0.847035189263436, 'penalty': 'l1', 'solver': 'liblinear'}
    knn_hp = {'metric': 'manhattan', 'n_neighbors': 4, 'p': 1.623422152178822,
              'weights': 'distance'}  # {'weights': 'uniform', 'p': 2, 'n_neighbors': 1, 'metric': 'manhattan'}
    svc_hp = {'C': 356.7134399554886, 'degree': 4, 'gamma': 0.1861718620917028,
              'kernel': 'rbf'}  # {'C': 38.83569559468877, 'degree': 6, 'kernel': 'rbf'}

    # default classifiers
    def_lr = LogisticRegression()
    def_knn = KNeighborsClassifier()
    def_svc = SVC()
    default_clf = [def_lr, def_knn, def_svc]

    # tuned classifiers
    lr_fs = LogisticRegression().set_params(**lr_hp)
    knn_fs = KNeighborsClassifier().set_params(**knn_hp)
    svc_fs = SVC().set_params(**svc_hp)
    tuned_clf = [lr_fs, knn_fs, svc_fs]
    for clf in tuned_clf:
        print(clf)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        print(f'\tAccuracy: {accuracy:.3f}')
        print(f'\tF1 Score: {f1:.3f}')
        print(clf, 'done\n')
    print('--------------------------------------------------------------------------------')
    print('------------------------------------with FS------------------------------------\n\n')
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=123)
    cv = StratifiedKFold(n_splits=10, random_state=123, shuffle=True)

    fitness_evol = []
    cxs = [0.7]
    logging.info('defaults, rnd_seed=100, noise 0-100 then scale, tourn=5, pop=200, 1xCV w shuffle')
    for clf in default_clf:
        for cxpb in cxs:
            accuracies = []
            f1_scores = []
            print('--------------------------------------------------------------------------------')
            logging.info('--------------------------------------------------------------------------------')
            print('clf:', clf)
            logging.info(clf)
            logging.info(clf.get_params())
            logging.info(f'cx_prob: {cxpb}')
            for i in range(1, 2):
                print('--------------------------------------------------------------------------------')
                print('run #', i)
                logging.info(f'run {i}')
                ea_estm = GAFeatureSelectionCV(
                    estimator=clf,
                    cv=cv,
                    scoring='f1',
                    generations=1,  # 40,
                    population_size=200,
                    crossover_probability=cxpb,
                    mutation_probability=1 / len(data.columns),
                    verbose=True,
                    elitism=True,
                    keep_top_k=1,
                    n_jobs=-1,
                    tournament_size=5
                )

                print(clf, 'with FS')
                print('params: :', clf.get_params())

                ea_estm.fit(X_train.values, y_train.values, callbacks=ProgressBar())
                sel_feats = ea_estm.get_feature_names_out(list(X))
                logging.info(len(sel_feats))
                logging.info(sel_feats)
                print(ea_estm.n_features, 'total features seen')
                print(len(sel_feats), 'features taken')
                print('\nbest_feat:', sel_feats)

                save_ga_results(ea_estm.history, i, f'{exp_name}_fitness_res_{clf}.csv')

                # -------training done-now eval------- #
                # X_test = ea_estm.transform(X_test)
                y_pred = ea_estm.predict(X_test)  # X_test.values

                acc = accuracy_score(y_pred, y_test)
                accuracies.append(acc)
                print('acc:', acc)
                logging.info(f'acc: {acc}')
                logging.info('acc:')
                logging.info(acc)
                f1 = f1_score(y_pred, y_test)
                f1_scores.append(f1)
                print('f1:', f1)
                logging.info(f'f1: {f1}')
                logging.info(f1)

                fitness_evol.append(ea_estm)

            print(clf)
            print('mean acc:', np.mean(accuracies))
            mean_f1 = np.mean(f1_scores)
            print('mean f1:', mean_f1)

    # for i in fitness_evol:
    #     plot_fitness_evolution(i)
    #
    # plt.show()


if __name__ == '__main__':
    main()
