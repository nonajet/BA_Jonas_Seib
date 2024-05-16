import csv
from pprint import pprint

import numpy as np


def write_to_csv(csvpath, features):
    csv_path = csvpath

    relevant_features = get_rel_attr_as_dict(features)  # filters relevant features and returns them as dict
    relevant_features = dict(sorted(relevant_features.items()))
    fields = list(relevant_features.keys())

    with open(csv_path, 'r+', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        csv_as_str = ''.join(','.join(row) for row in reader)
        if not csv_as_str or not csv.Sniffer().has_header(csv_as_str):  # TODO: always lazy eval?
            writer = csv.DictWriter(csvfile, fieldnames=fields, delimiter=',')
            writer.writeheader()

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields, delimiter=',')
        pprint(relevant_features)
        writer.writerows([relevant_features])


def get_rel_attr_as_dict(feature_container):  # only copy features relevant for later feature selection
    features = {}
    relevant = ['contact_durations',  # only iterable features
                'air_durations',
                'step_lengths',
                'peak_pressure',
                'avg_pres',
                'paw_area',
                'ratio_pres2area',
                'asc_peak_slopes',
                'des_peak_slopes',
                'vert_forces',
                'impulses']

    for attr, val in vars(feature_container).items():
        if attr in relevant:
            for paw_val in val:  # for every paw (key or pair)
                key = attr + '_' + paw_val  # save feature for each paw
                features[key] = np.round(np.median(val[paw_val]), 3)  # allow for each paw only one number
                key = 'std_' + key
                features[key] = np.round(np.std(val[paw_val]), 3)

    features['dog_id'] = feature_container.dog_id
    features['pace'] = feature_container.pace

    try:
        features['steps_A'] = feature_container.no_of_steps['A']  # Todo: temporary
        features['steps_C'] = feature_container.no_of_steps['C']
    except KeyError as ke:
        print(ke.args)

    return features


def get_attr_as_list(features):
    fields = []
    for attr in features:
        fields.append(attr)
    return fields
