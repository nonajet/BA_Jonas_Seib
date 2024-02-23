import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

a = np.arange(36).reshape(6, 6)

b = np.array([[0, 0, 0, 0, 0, 0],
              [0, 7, 8, 9, 10, 0],
              [0, 13, 14, 15, 16, 0],
              [1, 19, 20, 21, 22, 0],
              [1, 25, 26, 27, 28, 0],
              [1, 0, 0, 0, 0, 0]])

data = load_iris()
X, y = data["data"], data["target"]

A = [65, 61, 63, 45, 40, 1]
if any(val <= 3 for val in A):
    print(True)

if __name__ == '__main__':
    print(bool(''))
