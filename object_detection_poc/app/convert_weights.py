import numpy as np
import tensorflow as tf

from object_detection_poc.utils import config_util
from object_detection_poc.topologies.models import YoloV3Model


def convert_weights(conf):
    # TODO: Implement layer logic similar to https://github.com/xiaochus/YOLOv3/blob/master/yad2k.py
    darknet_weights_path = conf.DARKNET_WEIGHTS_PATH
    darknet_cfg_path = conf.DARKNET_CFG_PATH
    output_weights_dir = conf.WEIGHTS_DIR

    print(darknet_cfg_path, darknet_weights_path, output_weights_dir)

    yolo = YoloV3Model(conf)

    yolo_layer_names = yolo.model_layers.__dict__.keys()
    # print(yolo_layer_names)
    for yolo_layer_name in yolo_layer_names:
        yolo_layer = yolo.model_layers.__dict__[yolo_layer_name]
        print(yolo_layer)
        # yolo_layer.build(tf.TensorShape([None, 256, 256, 3]))
        # yolo_layer.set_weights([np.ones((3,3,3,32)), np.ones((32,))])


if __name__ == '__main__':
    conf = config_util.get_user_conf()
    convert_weights(conf)
