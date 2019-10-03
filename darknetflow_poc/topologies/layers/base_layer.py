import tensorflow as tf


class BaseLayer(tf.keras.layers.Layer):
    """
    Base class for implementing custom layers
    """
    def __init__(self):
        pass

    def build(self):
        pass

    def call(self, inputs):
        pass
