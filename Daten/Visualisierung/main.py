import logging

import mylib
from Daten.Visualisierung.paw_visualization import visualize
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
    filename = r'T0316445 Trab.xml'
    walk_id = '1'

    logger = logging.getLogger()  # logger.create_logger(dog_ident.replace(".xml", ".log"))
    abs_path, gait_id = set_params(walk_id, filename)
    visualize(abs_path, gait_id, visuals=False, vis_from=0, mx_start=0, total_view=False)
    feature_creation.paw_validation()
    feature_creation.calc_features()

    logger.info('####################################\nmovement (%s) '
                'finished\n####################################' % gait_id)
