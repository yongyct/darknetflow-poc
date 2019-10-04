import tensorflow as tf

from darknetflow_poc.topologies.models.base_model import BaseModel

from darknetflow_poc.utils.constants import PAD_SAME, RELU, SIGMOID


class TomModel(BaseModel):
    """
    Sample network
    """
    def __init__(self, conf):
        super(TomModel, self).__init__(conf)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=3,
            padding=PAD_SAME,
            activation=RELU
        )
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=2,
            strides=2
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=5,
            kernel_size=3,
            padding=PAD_SAME,
            activation=RELU
        )
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=2,
            strides=2
        )
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=300,
            activation=RELU
        )
        self.dropout1 = tf.keras.layers.Dropout(
            rate=0.4
        )
        self.dense2 = tf.keras.layers.Dense(
            units=150,
            activation=RELU
        )
        self.dropout2 = tf.keras.layers.Dropout(
            rate=0.4
        )
        self.logits = tf.keras.layers.Dense(
            units=4,
            activation=SIGMOID
        )
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=None):
        inputs = self.conv1(inputs)
        inputs = self.pool1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.pool2(inputs)
        inputs = self.flat(inputs)
        inputs = self.dense1(inputs)
        inputs = self.dropout1(inputs, training)
        inputs = self.dense2(inputs)
        inputs = self.dropout2(inputs, training)
        inputs = self.logits(inputs)
        return self.softmax(inputs)
