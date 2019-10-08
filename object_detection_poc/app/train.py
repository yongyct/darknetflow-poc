import os
import sys
import logging
import math
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import tensorflow as tf

from object_detection_poc.utils import config_util, validation_util, data_util
from object_detection_poc.exceptions.conf_error import InvalidConfigError
from object_detection_poc.topologies.models.tom_model import TomModel


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

    input_data_filenames = data_util.get_input_images(conf)
    batch_size = min(conf.BATCH_SIZE, len(input_data_filenames))
    n_batches = math.ceil(len(input_data_filenames) / batch_size)
    pool = ThreadPool()

    tom = TomModel(conf)
    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=tom)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, conf.WEIGHTS_DIR, max_to_keep=3)
    ckpt.restore(ckpt_mgr.latest_checkpoint)

    if ckpt_mgr.latest_checkpoint:
        print('Restored model from checkpoint: {}'.format(ckpt_mgr.latest_checkpoint))
    else:
        print('Training from Scratch...')

    if conf.USE_GPU:
        device = tf.device('/GPU:0')
    else:
        device = tf.device('/device:CPU:0')

    with device:
        for epoch in range(conf.N_EPOCHS):
            epoch_start_time = time.time()
            for i in range(n_batches):

                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                batch_data_filenames = input_data_filenames[start_idx:end_idx]
                batch_input, _ = pool.map(
                    data_util.get_preprocessed_image,
                    batch_data_filenames
                )
                batch_label = tf.convert_to_tensor(np.array(
                    [
                        float(batch_data_filename.split('\\')[-1].split('.')[0]) - 1
                        if '5' not in batch_data_filename else 0.
                        for batch_data_filename in batch_data_filenames
                    ]
                ))[None, :]

                loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

                with tf.GradientTape() as tape:
                    batch_output = tf.convert_to_tensor(tom(np.concatenate(batch_input, axis=0), training=True))
                    loss = loss_object(batch_label, batch_output)
                gradients = tape.gradient(loss, tom.trainable_variables)
                optimizer.apply_gradients(zip(gradients, tom.trainable_variables))

                ckpt.step.assign_add(1)
                if int(ckpt.step) % conf.SAVE_INTERVAL == 0:
                    save_path = ckpt_mgr.save()
                    print('Loss={}'.format(loss))
                    print('Preds={}'.format(batch_output))
                    print('Saving checkpoint: {}'.format(save_path))
            epoch_end_time = time.time()
            if (epoch + 1) % 10 == 0:
                logging.info('Time taken for epoch {}: {}'.format(epoch + 1, epoch_end_time - epoch_start_time))


if __name__ == '__main__':
    main()
