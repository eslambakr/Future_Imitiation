import tensorflow as tf
from tabulate import tabulate


def print_network_state():
    total_parameters = 0
    layer_list = []
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        name = variable.name
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        layer_list.append([name, shape, variable_parameters])
    layer_list.append(['total parameters', '===================', total_parameters])
    print(tabulate(layer_list, headers=['layer name', 'layer shape', 'total parameters'], tablefmt='orgtbl'))
