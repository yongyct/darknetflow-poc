import time
from keras import backend
import tensorflow as tf
from tensorflow.python.client import device_lib

backend.tensorflow_backend._get_available_gpus()

print('Device 0:\n', device_lib.list_local_devices()[0])
print('Device 1:\n', device_lib.list_local_devices()[1])

start_time = time.time()

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
timer = time.time()
print(timer - start_time)
