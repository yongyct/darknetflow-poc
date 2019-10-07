import os
import sys
import logging
import math
import time
from multiprocessing.pool import ThreadPool
from functools import partial

import numpy as np
import tensorflow as tf

from darknetflow_poc.utils import config_util, validation_util, data_util
from darknetflow_poc.exceptions.conf_error import InvalidConfigError
from darknetflow_poc.topologies.models import YoloV3Model


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

    yolo = YoloV3Model(conf)
    step = tf.Variable(0)
    ckpt = tf.train.Checkpoint(step=step, model=yolo)
    if tf.train.latest_checkpoint(conf.WEIGHTS_DIR):
        ckpt.restore(tf.train.latest_checkpoint(conf.WEIGHTS_DIR)).expect_partial()
        print('Predicting from checkpoint {}'.format(int(ckpt.step)))
    else:
        print('No checkpoint models found... predicting from random weights')

    if conf.USE_GPU:
        device = tf.device('/GPU:0')
    else:
        device = tf.device('/device:CPU:0')

    with device:

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_data_list = input_data_list[start_idx:end_idx]
            batch_input, _ = pool.map(
                partial(data_util.get_preprocessed_image, dim=conf.INPUT_DIM),
                batch_data_list
            )

            start_time = time.time()

            batch_output = yolo(np.concatenate(batch_input, axis=0), training=False)
            # TODO: implement post processing logic
            # print(batch_output)

            end_time = time.time()

            logging.info('Time taken: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()
