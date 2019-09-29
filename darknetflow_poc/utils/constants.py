"""
Module to keep program wide constants

- Constants for config keys to be in sync with
User Json Config schema if there are changes
"""

# Top Level Config Keys
JOB_CONF_KEY = 'job_conf'
MODEL_CONF_KEY = 'model_conf'

# 2nd Level Config Keys, job
DATA_DIR_KEY = 'data_dir'
BATCH_SIZE_KEY = 'batch_size'
USE_GPU_KEY = 'use_gpu'

# 2nd Level Config Keys, model
HEIGHT_KEY = 'height'
WIDTH_KEY = 'width'
CHANNELS_KEY = 'channels'

# Tensorflow Constants
PAD_VALID = 'VALID'
PAD_SAME = 'SAME'

# Others
IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
INPUT_NAME = 'input'
