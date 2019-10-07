import json
import argparse
import logging

from darknetflow_poc.utils.constants import JOB_CONF_KEY, MODEL_CONF_KEY, IN_DATA_DIR_KEY, BATCH_SIZE_KEY, \
    HEIGHT_KEY, WIDTH_KEY, CHANNELS_KEY, USE_GPU_KEY, WEIGHTS_DIR_KEY, N_EPOCHS_KEY, SAVE_INTERVAL_KEY, \
    N_CLASSES_KEY, KERAS_WEIGHTS_PATH_KEY, DARKNET_CFG_PATH_KEY, DARKNET_WEIGHTS_PATH_KEY, USE_WEBCAM_KEY, \
    LABELS_PATH_KEY, OBJECT_THRESHOLD_KEY, NMS_THRESHOLD_KEY, OUT_DATA_DIR_KEY, OPS_KEY

def get_user_conf():
    """
    Sets logging configurations + retrieve config from user provided json
    :return: UserConfig object containing user config
    """
    # Logging configs
    logging.basicConfig(level=getattr(logging, 'INFO', logging.INFO))

    # Program args
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', nargs='?', required=True, help='filename of json config file')
    program_args = parser.parse_args()

    logging.info('Json config provided: {}'.format(program_args.filename))

    return UserConfig(get_json_conf(program_args.filename))


def get_json_conf(filename):
    """
    Parses json information in the filename, and return it as a dictionary
    :param filename: absolute filename of user json config file
    :return: dictionary from json configuration
    """
    with open(filename) as json_file:
        conf_dict = json.load(json_file)
    return conf_dict


class UserConfig:
    """
    Object to hold configuration values
    """
    def __init__(self, conf):

        self.OPS = conf[JOB_CONF_KEY][OPS_KEY]
        self.IN_DATA_DIR = conf[JOB_CONF_KEY][IN_DATA_DIR_KEY]
        self.OUT_DATA_DIR = conf[JOB_CONF_KEY][OUT_DATA_DIR_KEY]
        self.WEIGHTS_DIR = conf[JOB_CONF_KEY][WEIGHTS_DIR_KEY]
        self.BATCH_SIZE = conf[JOB_CONF_KEY][BATCH_SIZE_KEY]
        self.N_EPOCHS = conf[JOB_CONF_KEY][N_EPOCHS_KEY]
        self.SAVE_INTERVAL = conf[JOB_CONF_KEY][SAVE_INTERVAL_KEY]
        self.USE_GPU = conf[JOB_CONF_KEY][USE_GPU_KEY]
        self.KERAS_WEIGHTS_PATH = conf[JOB_CONF_KEY][KERAS_WEIGHTS_PATH_KEY]
        self.DARKNET_WEIGHTS_PATH = conf[JOB_CONF_KEY][DARKNET_WEIGHTS_PATH_KEY]
        self.DARKNET_CFG_PATH = conf[JOB_CONF_KEY][DARKNET_CFG_PATH_KEY]
        self.USE_WEBCAM = conf[JOB_CONF_KEY][USE_WEBCAM_KEY]
        self.LABELS_PATH = conf[JOB_CONF_KEY][LABELS_PATH_KEY]
        self.OBJECT_THRESHOLD = conf[JOB_CONF_KEY][OBJECT_THRESHOLD_KEY]
        self.NMS_THRESHOLD = conf[JOB_CONF_KEY][NMS_THRESHOLD_KEY]

        self.INPUT_DIM = [
            conf[MODEL_CONF_KEY][HEIGHT_KEY],
            conf[MODEL_CONF_KEY][WIDTH_KEY],
            conf[MODEL_CONF_KEY][CHANNELS_KEY]
        ]
        self.N_CLASSES = conf[MODEL_CONF_KEY][N_CLASSES_KEY]
