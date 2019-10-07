"""
Module to keep program wide constants

- Constants for config keys to be in sync with
User Json Config schema if there are changes
"""

# Top Level Config Keys
JOB_CONF_KEY = 'job_conf'
MODEL_CONF_KEY = 'model_conf'

# 2nd Level Config Keys, job
OPS_KEY = 'ops'
IN_DATA_DIR_KEY = 'in_data_dir'
OUT_DATA_DIR_KEY = 'out_data_dir'
WEIGHTS_DIR_KEY = 'weights_dir'
BATCH_SIZE_KEY = 'batch_size'
USE_GPU_KEY = 'use_gpu'
N_EPOCHS_KEY = 'n_epochs'
SAVE_INTERVAL_KEY = 'save_interval'
KERAS_WEIGHTS_PATH_KEY = 'keras_weights_path'
DARKNET_WEIGHTS_PATH_KEY = 'darknet_weights_path'
DARKNET_CFG_PATH_KEY = 'darknet_cfg_path'
USE_WEBCAM_KEY = 'use_webcam'
LABELS_PATH_KEY = 'labels_path'
OBJECT_THRESHOLD_KEY = 'object_threshold'
NMS_THRESHOLD_KEY = 'nms_threshold'

# 2nd Level Config Keys, model
HEIGHT_KEY = 'height'
WIDTH_KEY = 'width'
CHANNELS_KEY = 'channels'
N_CLASSES_KEY = 'n_classes'

# Tensorflow Constants
PAD_VALID = 'valid'
PAD_SAME = 'same'
RELU = 'relu'
SIGMOID = 'sigmoid'

# Others
IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
INPUT_NAME = 'input'
PREDICT = 'predict'
