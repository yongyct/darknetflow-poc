import json
import argparse
import logging

from darknetflow_poc.utils.constants import JOB_CONF_KEY, MODEL_CONF_KEY, DATA_DIR_KEY, BATCH_SIZE_KEY, \
    HEIGHT_KEY, WIDTH_KEY, CHANNELS_KEY


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

        self.DATA_DIR = conf[JOB_CONF_KEY][DATA_DIR_KEY]
        self.BATCH_SIZE = conf[JOB_CONF_KEY][BATCH_SIZE_KEY]

        self.INPUT_DIM = [
            conf[MODEL_CONF_KEY][HEIGHT_KEY],
            conf[MODEL_CONF_KEY][WIDTH_KEY],
            conf[MODEL_CONF_KEY][CHANNELS_KEY]
        ]
