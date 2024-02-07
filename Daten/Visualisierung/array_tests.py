from matplotlib import pyplot as plt

import numpy as np
from skimage import measure
from skimage.draw import disk

a = np.arange(36).reshape(6, 6)

b = np.array([[0, 0, 0, 0, 0, 0],
              [0, 7, 8, 9, 10, 0],
              [0, 13, 14, 15, 16, 0],
              [0, 19, 20, 21, 22, 0],
              [0, 25, 26, 27, 28, 0],
              [0, 0, 0, 0, 0, 0]])

list1 = [1, 2, 3, 4, 5]
list2 = [3, 4, 5, 6, 7]


neue_daten = ['Max', 25, 'Berlin', {'zusatz_info': 'Beispiel', 'status': 'aktiv'}]
liste_fuer_csv = neue_daten[:-1] + [neue_daten[-1]['zusatz_info'], neue_daten[-1]['status']]
print(liste_fuer_csv)

if __name__ == '__main__':
    pass
