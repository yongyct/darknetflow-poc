import os
import cv2
import numpy as np

from darknetflow_poc.utils.constants import IMAGE_EXTENSIONS


def get_input_images(conf):
    """
    Get valid input files in provided data directory within json config
    :param conf: user provided json config
    :return: list of valid image files that can be used
    """
    data_dir = conf.DATA_DIR
    return [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)
            if file_name.split('.')[-1].lower() in IMAGE_EXTENSIONS]


def get_preprocessed_image(image):
    """
    Pre-process input image and returns it as a numpy ndarray
    :param image: input image to be preprocessed
    :return: numpy representation of image
    """
    img_norm = cv2.imread(image) / 255
    return np.expand_dims(a=img_norm, axis=0)
