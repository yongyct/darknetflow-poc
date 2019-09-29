import numpy as np
import tensorflow as tf

from darknetflow_poc.models.networks.basenet import BaseNet

from darknetflow_poc.utils.constants import INPUT_NAME, PAD_VALID, PAD_SAME


class TomNet(BaseNet):
    """
    Sample network
    """
    def __init__(self, conf):
        super().__init__(conf)

        # padding by sample, height, width, channel
        # Depending on NCHW or NHWC
        # pad1 = [[0, 0], [1, 1], [1, 1], [0, 0]]

        input = tf.placeholder(dtype=tf.float32, shape=[None] + conf.INPUT_DIM, name=INPUT_NAME)

        conv1 = tf.layers.conv2d(
            inputs=input,
            filters=32,
            kernel_size=3,
            strides=(1, 1),
            padding=PAD_VALID,
            activation=tf.nn.relu
        )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=2,
            strides=2
        )
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=5,
            kernel_size=3,
            padding=PAD_SAME,
            activation=tf.nn.relu
        )
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=2,
            strides=2
        )
        pool2_flat = tf.reshape(pool2, [-1, np.prod(pool2.get_shape()[1:])])
        dense1 = tf.layers.dense(
            inputs=pool2_flat,
            units=300,
            activation=tf.nn.relu
        )
        dense2 = tf.layers.dense(
            inputs=dense1,
            units=150,
            activation=tf.nn.relu
        )
        logits = tf.layers.dense(
            inputs=dense2,
            units=10
        )
        softmax = tf.nn.softmax(logits)

        self.input = input
        self.output = softmax
