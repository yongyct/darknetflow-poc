import numpy as np
import tensorflow as tf


class YoloV3Layer(tf.keras.layers.Layer):
    """
    YOLO v3 prediction layer implementation
    """
    def __init__(self, anchors, conf):
        super(YoloV3Layer, self).__init__()
        self.anchors = anchors
        self.conf = conf

    def call(self, inputs):
        # TODO: Validation -> N_CLASSES + 5 * n_anchors == YOLO_INPUT.shape[-1]
        batch_size = inputs.shape[0]
        grid_size = inputs.shape[2]
        strides = self.conf.INPUT_DIM[1] // grid_size
        n_anchors = len(self.anchors)
        n_attrs_per_bbox = 5 + self.conf.N_CLASSES

        # Reshaping inputs
        inputs = tf.reshape(inputs, (batch_size, n_attrs_per_bbox*n_anchors, grid_size**2))
        inputs = tf.transpose(inputs, (0, 2, 1))
        inputs = tf.reshape(inputs, (batch_size, (grid_size**2)*n_anchors, n_attrs_per_bbox))

        # Sigmoid centre_x, centre_y, object confidence
        ## inputs[:, :, 0] = tf.math.sigmoid(inputs[:, :, 0])
        ## inputs[:, :, 1] = tf.math.sigmoid(inputs[:, :, 1])
        ## inputs[:, :, 4] = tf.math.sigmoid(inputs[:, :, 4])
        inputs = tf.tensor_scatter_nd_update(
            inputs, self._get_indices(inputs, 0),
            self._get_updates(tf.math.sigmoid(inputs[:, :, 0]))
        )
        inputs = tf.tensor_scatter_nd_update(
            inputs, self._get_indices(inputs, 1),
            self._get_updates(tf.math.sigmoid(inputs[:, :, 1]))
        )
        inputs = tf.tensor_scatter_nd_update(
            inputs, self._get_indices(inputs, 4),
            self._get_updates(tf.math.sigmoid(inputs[:, :, 4]))
        )

        # Add centre offsets
        grid_axis = np.arange(grid_size)
        a, b = np.meshgrid(grid_axis, grid_axis)
        x_offset = tf.reshape(tf.convert_to_tensor(a, dtype=tf.float32), (-1, 1))
        y_offset = tf.reshape(tf.convert_to_tensor(b, dtype=tf.float32), (-1, 1))
        xy_offset = tf.expand_dims(
            tf.reshape(
                tf.tile(
                    tf.concat([x_offset, y_offset], axis=1),
                    [1, n_anchors]
                ),
                (-1, 2)
            ),
            axis=0
        )
        ## inputs[:, :, :2] += xy_offset
        inputs = tf.tensor_scatter_nd_update(
            inputs, self._get_indices(inputs, slice(0, 2)),
            self._get_updates(tf.add(inputs[:, :, :2], xy_offset[:, :, :2]))
        )

        # 'Normalize' the anchors first, to denormalize at the end
        anchors = [(anchor[0] / strides, anchor[1] / strides) for anchor in self.anchors]
        anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
        # log space transform anchor dims, applied to width & height of bbox pred
        anchors = tf.expand_dims(tf.tile(anchors, [grid_size**2, 1]), axis=0)
        ## inputs[:, :, 2:4] = tf.math.multiply(tf.math.exp(inputs[:, :, 2:4]), anchors)
        inputs = tf.tensor_scatter_nd_update(
            inputs, self._get_indices(inputs, slice(2, 4)),
            self._get_updates(tf.math.multiply(tf.math.exp(inputs[:, :, 2:4]), anchors))
        )
        # Sigmoid the class predictions (after index 4)
        ## inputs[:, :, 5:n_attrs_per_bbox] = tf.math.sigmoid(inputs[:, :, 5:n_attrs_per_bbox])
        inputs = tf.tensor_scatter_nd_update(
            inputs, self._get_indices(inputs, slice(5, n_attrs_per_bbox)),
            self._get_updates(tf.math.sigmoid(inputs[:, :, 5:n_attrs_per_bbox]))
        )

        # TODO: Clean up item assignment when TF2.0 has better support for item assignment of EagerTensor

        # Denormalize and return bbox
        return tf.tensor_scatter_nd_update(
            inputs, self._get_indices(inputs, slice(0, 4)),
            self._get_updates(tf.math.multiply(inputs[:, :, :4], strides))
        )

    @staticmethod
    def _get_indices(tensor, n):
        indices = []
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                if isinstance(n, int):
                    indices.append([i, j, n])
                elif isinstance(n, slice):
                    start, stop, step = n.indices(tensor.shape[2])
                    for k in range(start, stop, step):
                        indices.append([i, j, k])
        return tf.convert_to_tensor(indices)

    @staticmethod
    def _get_updates(tensor):
        return tf.reshape(tensor, (-1,))
