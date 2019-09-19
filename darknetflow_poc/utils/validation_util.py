import logging

from darknetflow_poc.utils import data_util
from darknetflow_poc.exceptions.conf_error import InvalidConfigError


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
        raise InvalidConfigError('No valid images/videos found in path: {}'.format(conf.DATA_DIR))
