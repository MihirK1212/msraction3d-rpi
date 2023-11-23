MSR_ACTION_3D_DATA_DIR = './msraction3d'

STATISTICAL_MOMENTS = 'statistical_moments'
WINDOW_DIVISION = 'window_division'

USE_AGGREGATION_METHODS = [
    STATISTICAL_MOMENTS, 
    WINDOW_DIVISION
]

NUM_JOINTS = 20
NUM_CLASSES = 20


TRAINER_SAVE_DIR = './trainers'
TRAINING_CONFIG_PATH = TRAINER_SAVE_DIR + '/training_config.json'

X_TRAIN_KEY = 'X_train'
X_VAL_KEY = 'X_val'
Y_TRAIN_KEY = 'y_train'
Y_VAL_KEY = 'y_val'

