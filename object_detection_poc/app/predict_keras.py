import os
import sys
import logging
import time
import cv2

import tensorflow as tf

from object_detection_poc.utils import config_util, validation_util, data_util
from object_detection_poc.exceptions.conf_error import InvalidConfigError


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

    yolo = tf.keras.models.load_model(conf.KERAS_WEIGHTS_PATH)

    if conf.USE_GPU:
        device = tf.device('/GPU:0')
    else:
        device = tf.device('/device:CPU:0')

    with device:

        start_time = time.time()
        for input_data_filename in input_data_list:
            original_image = cv2.imread(input_data_filename)
            input_data = data_util.get_preprocessed_image(input_data_filename, conf.INPUT_DIM)
            output_data = yolo(input_data)
            # TODO: implement post processing logic
            if not conf.USE_WEBCAM:
                boxes, classes, scores = data_util.yolo_out(
                    outs=output_data,
                    shape=original_image.shape,
                    conf=conf
                )
                if boxes is not None:
                    data_util.draw_bounding_boxes(
                        image=original_image,
                        boxes=boxes,
                        scores=scores,
                        classes=classes,
                        all_classes=data_util.get_all_class_labels(conf)
                    )
                    cv2.imwrite(os.path.join(conf.OUT_DATA_DIR, os.path.basename(input_data_filename)), original_image)

        end_time = time.time()

        logging.info('Time taken: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()
