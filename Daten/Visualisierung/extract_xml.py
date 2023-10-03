import xml.etree.ElementTree as et
import numpy as np


def extract_matrix(filepath):
    root = et.parse(filepath).getroot()

    # get No. of matrices in one data set
    for attr in root.iter('count'):
        count = int(attr.text)
        print(count)

    # extract matrix data
    quant_elements = root.findall(".//data/quant")
    matrices = []
    for quant_element in quant_elements:
        quant_data = {  # ab hier Chat-GPT Magie bis Zeile 21
            "cells": [
                list(map(float, cell_data.split()))
                for cell_data in quant_element.find("cells").text.splitlines()
            ],
        }
        matrices.append(quant_data)

    data = []  # final list with matrices

    for matrix in matrices:
        # print('matrix: ', matrix)
        tmp_matrix = []
        for rows in matrix['cells']:
            if len(rows) >= 1:
                tmp_matrix.append(rows)
        data.append(tmp_matrix)

    print(data)  # so etwa sollen die Daten vorliegen: eine Liste, die mehrere Matrizen (unterschiedl. Größe) enthält
    # print('----------------------------------------new data set----------------------------------------')
    # view_mx = np.array(data)
    # print(view_mx)


if __name__ == '__main__':
    extract_matrix(r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Rohdaten\T0307068 Schritt.xml')
