import sys
import logging
import math
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import tensorflow as tf

from darknetflow_poc.utils import config_util, validation_util, data_util
from darknetflow_poc.exceptions.conf_error import InvalidConfigError
from darknetflow_poc.models.networks.tomnet import TomNet


def handle_error(e):
    """
    Error handling during the prediction dataflow
    """
    logging.error(str(e) + '\n...Exiting program...')
    sys.exit(0)


def main():
    """
    Main data flow for prediction operations
    :return: None
    """
    conf = config_util.get_user_conf()
    try:
        validation_util.validate_user_conf(conf)
    except InvalidConfigError as e:
        handle_error(e)

    input_data_list = data_util.get_input_images(conf)
    batch_size = min(conf.BATCH_SIZE, len(input_data_list))
    n_batches = math.ceil(len(input_data_list) / batch_size)
    pool = ThreadPool()

    tom = TomNet(conf)

    if conf.USE_GPU:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
    else:
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
    gv_init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(gv_init)

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_data_list = input_data_list[start_idx:end_idx]
            batch_input = pool.map(
                data_util.get_preprocessed_image,
                batch_data_list
            )

            feed_dict = {tom.input: np.concatenate(batch_input, axis=0)}

            start_time = time.time()
            batch_output = sess.run(tom.output, feed_dict=feed_dict)

            # TODO: implement post processing logic
            logging.info(batch_output)

            end_time = time.time()

            logging.info('Time taken: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()
