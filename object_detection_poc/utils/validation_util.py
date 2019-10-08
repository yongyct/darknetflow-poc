import os
import logging

from object_detection_poc.utils import data_util
from object_detection_poc.exceptions.conf_error import InvalidConfigError

from object_detection_poc.utils.constants import PREDICT


def validate_user_conf(conf):
    """
    Validate the user provided json configuration during runtime
    :param conf: user provided json config
    :return: None
    """
    logging.info('Validating user config...')
    logging.info(conf)

    # TODO: Add validation logic
    # Input data availability check
    is_no_input_data = len(data_util.get_input_images(conf)) == 0
    if is_no_input_data:
        raise InvalidConfigError('No valid images/videos found in path: {}'.format(conf.IN_DATA_DIR))

    # Create output folders if not available
    if conf.OPS.strip().lower() == PREDICT:
        if not os.path.exists(conf.OUT_DATA_DIR):
            os.mkdir(conf.OUT_DATA_DIR)
