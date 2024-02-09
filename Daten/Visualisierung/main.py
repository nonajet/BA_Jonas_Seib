import logging
import os
import warnings

import mylib
from Daten.Visualisierung import extract_xml
from Daten.Visualisierung.paw_visualization import visualize
from save_features_csv import write_to_csv


def set_logger(logname):
    _abs_path = mylib.dirname + '\\' + logname
    log = logging.getLogger()
    logging.basicConfig(filename=logname,
                        filemode='w',  # w:=truncate+write; a:=append
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    file_handler = logging.FileHandler(logname)
    file_handler.terminator = ""  # avoid blank lines in log
    file_handler.setLevel(logging.INFO)
    log.addHandler(file_handler)
    log.info('log initialised')
    return log


def valid_features(features):
    for steps in features.no_of_steps.value():
        if steps <= 1:
            return False
    return True


if __name__ == '__main__':
    csv_file = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\dog_features_data.csv'
    filename = ''  # r'T0454999 Trab.xml'  # T0391053, T0316445, T0454128,
    # filename = 0
    walk_id = '1'
    base_dir = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Rohdaten\Rohdaten_test'
    logger = set_logger('features.log')  # logger.create_logger(dog_ident.replace(".xml", ".log"))

    if filename:
        try:
            gait = 'gait_' + walk_id
            Features = visualize(os.path.join(base_dir, filename), gait,
                                 visuals=True, vis_from=420, mx_start=0,
                                 total_view=True)
            Features.calc_features()
        except UserWarning as uw:
            warnings.warn("'{}': unreliable data ({})".format(filename, uw.args))

    valid = 0
    invalid = 0
    for f in os.listdir(base_dir):
        f_path = os.path.join(base_dir, f)
        if os.path.isfile(f_path) and 'Trab' in f:
            measures = extract_xml.get_no_of_measures(f_path)
            for gait_id in range(1, measures + 1):  # gait numbering starts at 1
                gait_id = 'gait_' + str(gait_id)
                print('\n', f, ':', gait_id, 'trying')
                logger.info("{}: {} trying".format(f, gait_id))
                try:
                    if 'T0455570' in f_path:
                        print('T0455570')
                    Features = visualize(f_path, gait_id, visuals=False, vis_from=0, mx_start=0, total_view=False)
                    Features.calc_features()
                    write_to_csv(csv_file, Features)
                    valid += 1
                    print("{}: {} successful".format(f, gait_id))
                    logger.info("{}: {} successful".format(f, gait_id))

                except (IndexError, AttributeError) as iea:
                    invalid += 1
                    logger.exception('{}:'.format(f), iea.args)
                    warnings.warn("{} could not open file".format(f))
                    continue

                except UserWarning as uw:
                    invalid += 1
                    logger.exception('{}: {}'.format(f, uw.args))
                    warnings.warn("'{}' unreliable data: {}".format(f, uw.args))
                    continue

                except AssertionError as ae:
                    invalid += 1
                    logger.exception('{}: {}'.format(f, ae))
                    warnings.warn("'{}' assertion failed".format(f))
                    continue
