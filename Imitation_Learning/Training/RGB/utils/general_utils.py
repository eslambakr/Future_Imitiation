import tensorflow as tf
import numpy as np


def count_tf_trainable_params():
    all = 0
    for i in tf.trainable_variables():
        all += np.prod(i.get_shape().as_list())
    return all