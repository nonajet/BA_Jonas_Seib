import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import loguniform

if __name__ == '__main__':
    csv_file = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\dog_features_data.csv'
    base_dir = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Rohdaten'

    for i in range(1, 11):
        print(i)

    fig, ax = plt.subplots(1, 1)
    a, b = 0.01, 1.25
    x = np.linspace(loguniform.ppf(0.01, a, b), loguniform.ppf(0.99, a, b), 100)
    ax.plot(x, loguniform.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='loguniform pdf')
    plt.show()

    # rv = loguniform(a, b)
    # ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
