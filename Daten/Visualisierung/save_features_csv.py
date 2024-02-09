import csv
from pprint import pprint

import numpy as np


def write_to_csv(csvpath, features):
    csv_path = csvpath

    relevant_features = get_rel_attr_as_dict(features)  # filters relevant features and returns them as dict
    fields = get_attr_as_list(relevant_features)

    # with open(csv_path, 'r', newline='') as csvfile:  # add header if necessary
    #     first_k_lines = csvfile.readlines()[20:]
    #     lines = ''.join(first_k_lines)
    #     print('csv:', lines)
    #     sniffer = csv.Sniffer()
    #     header_needed = not sniffer.has_header(lines)

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields, delimiter=',')
        # if header_needed:
        csvfile.seek(0)  # start of file
        writer.writeheader()
        csvfile.seek(2)  # end of file

        pprint(relevant_features)
        writer.writerows([relevant_features])


def get_rel_attr_as_dict(feature_container):  # only copy features relevant for later feature selection
    features = {}
    relevant = ['contact_durations',  # only iterable features
                'air_durations',
                'step_lengths',
                'peak_pressure',
                'avg_pres',
                'paw_area_cm2',
                'ratio_pres2area',
                'asc_peak_slopes',
                'des_peak_slopes']

    for attr, val in vars(feature_container).items():
        if attr in relevant:
            for paw_val in val:
                key = attr + '_' + paw_val  # save feature for each paw
                features[key] = np.round(np.median(val[paw_val]), 3)  # allow for each paw only one number

    features['pace'] = feature_container.pace
    return features


def get_attr_as_list(features):
    fields = []
    for attr in features:
        fields.append(attr)
    return fields
