import tensorflow as tf
import tensorflow.keras.layers as layers

from darknetflow_poc.topologies.models.base_model import BaseModel

from darknetflow_poc.utils.constants import PAD_SAME, RELU, SIGMOID


class YoloV3Model(BaseModel):
    """
    Implementation of tf keras version of YOLO V3
    """
    def __init__(self, conf):
        super().__init__(conf)
        self.conv1 = layers.Conv2D(
            filters=32,
            kernel_size=3,
            stride=1,
            padding=PAD_SAME
        )
        self.bn = layers.BatchNormalization(
            epsilon=1e-5,
            momentum=0.999,  # either 1 or 0 to shut off effects
            center=False,
            scale=True
        )
        self.conv2 = layers.Conv2D(
            filters=64,
            kernel_size=3,
            stride=2,
            padding=PAD_SAME  # stride 2 pad 1, how to achieve?
        )
        self.conv3 = layers.Conv2D(
            filters=32,
            kernel_size=1,
            stride=1,
            padding=PAD_SAME
        )
        self.conv4 = layers.Conv2D(
            filters=64,
            kernel_size=3,
            stride=1,
            padding=PAD_SAME  # stride 2 pad 1, how to achieve?
        )

    def call(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.bn(tf.nn.leaky_relu(features=inputs, alpha=0.1))
        inputs = self.conv2(inputs)
        inputs = self.bn(tf.nn.leaky_relu(features=inputs, alpha=0.1))
        inputs = self.conv3(inputs)
        inputs = self.bn(tf.nn.leaky_relu(features=inputs, alpha=0.1))
        inputs = self.conv4(inputs)
        inputs = self.bn(tf.nn.leaky_relu(features=inputs, alpha=0.1))
        return inputs
