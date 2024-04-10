import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

csv_path = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Datensätze\Trab_data.csv'
raw_df = pd.read_csv(csv_path, sep=';')
raw_df.drop(columns=['dog_id', 'level'], inplace=True)
# for i in range(1, 6):
#     raw_df[f'random_{i}'] = np.random.rand(raw_df.shape[0]) * 10

scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(raw_df), columns=raw_df.columns)

X = df.drop(columns='target')
y = raw_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
svc = SVC()
cv = StratifiedKFold(shuffle=True, random_state=42)
# select classifier
est = clf

# ffs_best = None
# for i in range(1, 24):
#     sfs_forward = SequentialFeatureSelector(
#         est,
#         direction="forward",
#         n_features_to_select=i,
#         cv=cv,
#         n_jobs=-1
#     )
#     sfs_forward.fit(X_train, y_train)
#     ffs_best = sfs_forward
#
# print('forward')
# print(len(ffs_best.support_), 'total features')
# print('len of df:', len(list(df)))
# sel_feats = ffs_best.get_feature_names_out(list(X))
# print('best_feat:', sel_feats, '(', len(sel_feats), ')')
# print('features:', ffs_best.support_)

# X_test = sfs_forward.transform(X_test)
# print('acc:', accuracy_score(y_test, y_pred))
