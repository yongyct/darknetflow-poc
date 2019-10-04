import tensorflow as tf

from darknetflow_poc.topologies.layers.base_layer import BaseLayer


class YoloV3Layer(BaseLayer):
    """
    Base class for implementing custom layers
    """
    def __init__(self, anchors):
        super(YoloV3Layer, self).__init__()
        self.anchors = anchors
        pass

    def build(self):
        pass

    def call(self, inputs):
        pass
