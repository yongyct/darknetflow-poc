import tensorflow as tf


class BaseModel(tf.keras.Model):
    """
    Base class for forming other network topologies
    """
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        print('Calling BaseNet')

    def call(self, inputs):
        pass
