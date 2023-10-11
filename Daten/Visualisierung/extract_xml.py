import xml.etree.ElementTree as et

import numpy as np
import matplotlib.pyplot as plt


def get_cell_count(filepath):  # not robustified yet
    root = et.parse(filepath).getroot()
    x = int(root[8][0].text)  # 9th child from root is cell_count
    y = int(root[8][1].text)
    return x, y


def get_count(filepath):  # get No. of matrices in one data set
    root = et.parse(filepath).getroot()
    for attr in root.iter('count'):
        count = int(attr.text)
        print(count)


def extract_matrices(filepath):
    root = et.parse(filepath).getroot()

    # extract matrix data from xml
    quant_elements = root.findall(".//data/quant")
    matrices = []  # temp. array for matrices from xml; may contain empty lists too
    for quant_element in quant_elements:
        quant_data = {
            "cells": [
                list(map(float, cell_data.split())) for cell_data in quant_element.find("cells").text.splitlines()
            ],
        }
        matrices.append(quant_data)

    data = []  # final list with matrices

    for matrix in matrices:  # remove empty lists from 'cells'-field/array
        tmp_matrix = []
        for rows in matrix['cells']:
            if len(rows) >= 1:
                tmp_matrix.append(rows)
        data.append(tmp_matrix)

    # print(data[0])
    # print('----------------------------------------new data set----------------------------------------')
    # np_matrix = np.array(data, dtype=object)

    return data


def global_max(matrix):  # not tested or used yet; used for finding global max in one or more matrices
    gmax = -1
    for mx in matrix:
        for row in mx:
            if max(row) > global_max:
                gmax = max(row)
    return gmax


def visualize(matrix):
    fig, ax = plt.subplots()
    plt.ion()

    for mx in matrix:
        # if mx == []
        # fig.colorbar(im, ax=ax, label='Interactive colorbar')

        mx_np = np.matrix(mx)
        ax.matshow(mx_np)
        plt.pause(0.01)
        ax.clear()

        # caxes = axes.matshow(mx, interpolation='nearest')
        # figure.colorbar(caxes)


if __name__ == '__main__':
    filep = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Rohdaten\test2.xml'
    a = extract_matrices(r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Rohdaten\test2.xml')
    # print(a)
    visualize(a)
    # print(get_cell_count(filep))
