import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

a = np.arange(36).reshape(6, 6)

b = np.array([[0, 0, 0, 0, 0, 0],
              [0, 7, 8, 9, 10, 0],
              [0, 13, 14, 15, 16, 0],
              [1, 19, 20, 21, 22, 0],
              [1, 25, 26, 27, 28, 0],
              [1, 0, 0, 0, 0, 0]])


class Test(object):
    def __init__(self):
        self.i = 0


if __name__ == '__main__':
    csv_file = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\dog_features_data.csv'
    base_dir = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Rohdaten'
