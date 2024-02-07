import csv
from pprint import pprint


def write_to_csv(csvpath, features):
    csv_path = csvpath

    fields = get_attr(features)

    with open(csv_path, 'w+', newline='') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=fields, delimiter=',')
        # writer.writeheader()
        pprint(features)
        writer.writerows([features])

    with open(csv_path, 'r+', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for line in reader:
            print(line)

        first_lines = csvfile.readline()
        for line in first_lines:
            print(line)
        print('\nfirst:\n', first_lines)
        # if first_lines and not csv.Sniffer().has_header(csvfile.readline()):


def get_attr(features):
    fields = []
    for attr in features:
        fields.append(attr)
    return fields
