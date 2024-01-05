# handling imports

# handling constants and flags
GAIT_TYPE = -1  # 0 == Schritt/walk; 1 == Trab/trot
DIRECTION = -1  # 0 == backward; 1 == forward;
three_paws = False  # false until dog has min. 3 paws on mat for first time; relevant for Schritt
two_paws = False  # false until dog has min. 3 paws on mat for first time; relevant for Trab
NEIGHBOR_DIST = 7

dirname = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Rohdaten'

# handling global variables, objects and funcs
current_dog_log = ''


def set_dog_id(abs_filepath):
    dog_ident_log = abs_filepath.split('\\')[-1].replace(".xml", ".log")  # os.path.basename(filep))
    global current_dog_log
    current_dog_log = dog_ident_log


def get_dog_id():
    if current_dog_log:
        return current_dog_log
