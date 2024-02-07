import logging
import os

import mylib
from Daten.Visualisierung import extract_xml
from Daten.Visualisierung.paw_visualization import visualize
from save_features_csv import write_to_csv
import feature_creation


def set_params(_walk_id, _filename):
    _abs_path = mylib.dirname + '\\' + _filename
    _gait_id = "gait_" + _walk_id
    mylib.set_dog_log(_abs_path)
    dog_ident = mylib.get_dog_log()

    logging.basicConfig(filename=dog_ident,
                        filemode='w',  # w:=truncate+write; a:=append
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger.info("start of '%s' with '%s'\n" % (dog_ident.replace(".log", ""), _gait_id))

    return _abs_path, _gait_id


if __name__ == '__main__':
    csv_file = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\dog_features_data.csv'
    filename = r'T0316445 Trab.xml'  # T0391053, T0316445
    walk_id = '2'
    base_dir = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Rohdaten'

    ctr = 0
    for f in os.listdir(base_dir):
        f_path = os.path.join(base_dir, f)
        if os.path.isfile(f_path) and 'Trab' in f:
            ctr += 1
            measures = extract_xml.get_no_of_measures(f_path)
            print(f, ': ', measures)
            for i in range(measures):
                pass
    print(ctr)

    logger = logging.getLogger()  # logger.create_logger(dog_ident.replace(".xml", ".log"))
    abs_path, gait_id = set_params(walk_id, filename)
    visualize(abs_path, gait_id, visuals=False, vis_from=0, mx_start=0, total_view=False)
    Features = feature_creation.calc_features()
    write_to_csv(csv_file, Features)
    logger.info('################\nmovement %s finished\n################' % gait_id)
