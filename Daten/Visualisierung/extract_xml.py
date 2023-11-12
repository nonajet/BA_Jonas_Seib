import xml.etree.ElementTree as et

import numpy as np
import matplotlib.pyplot as plt

NS = {'zb': 'http://www.zebris.de/measurements'}


def get_cell_count(filepath, id):  # not robustified yet
    movement = get_movement(filepath, id)
    cc_tuple = movement.find('.//zb:cell_count', NS).text
    x = int(cc_tuple[0])
    y = int(cc_tuple[1])
    return y, x


def get_count(filepath, id):  # get No. of matrices in one data set
    movement = get_movement(filepath, id)
    return int(movement.find('.//zb:count', NS).text)


def get_movement(filepath, id):
    root = et.parse(filepath).getroot()
    movements = root.findall('.//zb:movement', NS)
    for movement in movements:
        if movement.find('.//zb:id', NS).text == id:
            return movement


def extract_matrices(filepath, id):
    root = get_movement(filepath, id)

    # extract matrix data from xml
    quant_elements = root.findall(".//zb:data/zb:quant", NS)
    matrices = []  # temp. array for matrices from xml; may contain empty lists too
    cell_begin = []
    for quant_element in quant_elements:
        quant_data = {
            "cells": [
                list(map(float, cell_data.split())) for cell_data in
                quant_element.find("zb:cells", NS).text.splitlines()
            ],
        }
        matrices.append(quant_data)

        tmp_x = int(quant_element.find('.//zb:cell_begin/zb:x', NS).text)
        tmp_y = int(quant_element.find('.//zb:cell_begin/zb:y', NS).text)
        cell_begin.append([tmp_x, tmp_y])

    data = []  # final list with matrices

    for matrix in matrices:  # remove empty lists from 'cells'-field/array
        tmp_matrix = []
        for rows in matrix['cells']:
            if len(rows) >= 1:
                tmp_matrix.append(rows)
        data.append(tmp_matrix)

    return data, cell_begin


def visualize(filepath, id):
    matrix, offset = extract_matrices(filepath, id)
    fig, ax = plt.subplots(1, 3)
    plt.ion()

    ax_local = ax[0]
    ax_local.set_title('local')

    ax_global = ax[1]
    ax_global.set_title('global')

    ax_total = ax[2]
    total_mx = np.zeros((481, 64))

    plt.axis('off')

    mx_ctr = 0
    for mx in matrix:
        if mx and mx_ctr % 5 == 0:
            mx_np = np.array(mx)

            # local
            ax_local.imshow(
                np.flipud(np.fliplr(mx_np)))  # rotate 180° to fit vertical direction of matrix to global view
            ax_local.set_axis_off()

            # global
            global_mx = create_global_mx(mx_np, offset[mx_ctr])
            ax_global.matshow(global_mx)
            ax_global.set_xticks([])
            # ax_global.set_yticks([])

            # total (drains performance heavily)
            # total_mx += global_mx
            # ax_total.matshow(total_mx)

            plt.pause(0.0001)
        mx_ctr += 1

    print(
        '############################################\nmovement (%s) '
        'finished\n############################################' % id)


def create_global_mx(local_mx, offset):
    global_mx = np.zeros((481, 64))
    x_off = offset[1] - 1
    if x_off < 0: x_off = 0
    y_off = offset[0] - 1
    if y_off < 0: y_off = 0
    rows = local_mx.shape[0]
    cols = local_mx.shape[1]

    try:
        global_mx[-1 - x_off - rows:-1 - x_off, y_off:y_off + cols] = local_mx
    except ValueError as e:
        print(e)
    return global_mx


if __name__ == '__main__':
    filep = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Rohdaten\T0307068 Trab.xml'
    visualize(filep, 'gait_2')
