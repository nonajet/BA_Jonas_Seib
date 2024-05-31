import re
from collections import Counter
from pprint import pprint
import numpy as np


def extract_acc_values(log_string):
    acc_values = re.findall(r'acc: (\d+\.\d+)\n', log_string)
    return [float(value) for value in acc_values]


def extract_f1_values(log_string):
    acc_values = re.findall(r'f1: (\d+\.\d+)\n', log_string)
    return [float(value) for value in acc_values]


def average_acc(log_string):
    acc_values = extract_acc_values(log_string)
    return np.mean(acc_values)


def average_f1(log_string):
    f1_values = extract_f1_values(log_string)
    return np.mean(f1_values)


def extract_square_brackets(log_content):
    # Regular expression to find content within square brackets, including multiline
    bracket_pattern = re.compile(r'\[([^]]+)]')
    matches = bracket_pattern.findall(log_content)

    # Return the matches
    return matches


def find_clf(clf_log):
    if 'SVC' in clf_log:
        return 'svc'
    elif 'Logistic' in clf_log:
        return 'lr'
    elif 'KNeigh' in clf_log:
        return 'knn'
    else:
        return None


def parse_log_and_count_features(log):
    clf_logs = log.split("--------------------------------------------------------------------------------")[1:]

    for clf_log in clf_logs:
        clf_type = find_clf(clf_log)
        if clf_type:
            print(clf_type)
        sel_feats_brackets = extract_square_brackets(clf_log)
        all_feats = " ".join(sel_feats_brackets).replace('\n', ' ').replace(' ', ',').split(',')
        feature_counts = Counter(all_feats)
        pprint(feature_counts)

        print(average_acc(clf_log))
        print(average_f1(clf_log))


# Function to read log file content
def read_log_file(file_path):
    with open(file_path, 'r') as file:
        log_content = file.read()
    return log_content


# Main function to run the feature counting process
def main(log_file_path):
    log_content = read_log_file(log_file_path)
    parse_log_and_count_features(log_content)


# Example usage
if __name__ == "__main__":
    log_file_path = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\vw_fss3.log'
    main(log_file_path)
    print(log_file_path.split('\\')[-1])
