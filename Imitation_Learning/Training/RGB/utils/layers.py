import math
import numpy as np
import tensorflow as tf


# Summaries for variables
def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var:
    :return:
    """
    with tf.variable_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.variable_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# initialization for deconv layer
def get_deconv_filter(f_shape, l2_strength=1e-5):
    """
    The initializer for the bilinear convolution transpose filters
    :param f_shape: The shape of the filter used in convolution transpose.
    :param l2_strength: L2 regularization parameter.
    :return weights: The initialized weights.
    """
    width = f_shape[0]
    height = f_shape[0]
    f = math.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    # TODO investigate in this
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear
    # for i in range(f_shape[2]):
    #     for j in range(f_shape[3]):
    #         weights[:, :, i, j] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return _variable_with_weight_decay(weights.shape, init, l2_strength)


# Just Convolution 2d with it's weights and biases
def conv2d(name, x, output_dim, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer()):
    """
    Conv2d Wrapper for shab7ana
    :param name:
    :param x:
    :param output_dim:
    :param kernel_size:
    :param padding:
    :param stride:
    :param initializer:
    :return out, weights, biases:
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        with tf.variable_scope('weights'):
            w = tf.get_variable('w_conv', kernel_shape, tf.float32, initializer=initializer)
            variable_summaries(w)
        with tf.variable_scope('biases'):
            b = tf.get_variable('biases_conv', [output_dim], initializer=tf.constant_initializer(0.1))
            variable_summaries(b)
        with tf.variable_scope('conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, b)

    return out


# Just Dilated Convolution with it's weights and biases
def atrous_conv2d(name, x, output_dim, kernel_size=(3, 3), rate=2, padding='SAME',
                  initializer=tf.contrib.layers.xavier_initializer()):
    """
        Dilated Conv2d Wrapper for shab7ana
        :param name:
        :param x:
        :param output_dim:
        :param kernel_size:
        :param rate:
        :param padding:
        :param initializer:
        :return out, weights, biases:
        """
    with tf.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        with tf.variable_scope('weights'):
            w = tf.get_variable('w_conv', kernel_shape, tf.float32, initializer=initializer)
            variable_summaries(w)
        with tf.variable_scope('biases'):
            b = tf.get_variable('biases_conv', [output_dim], initializer=tf.constant_initializer(0.1))
            variable_summaries(b)
        with tf.variable_scope('atrous_conv2d'):
            conv = tf.nn.atrous_conv2d(x, w, rate, padding)
            out = tf.nn.bias_add(conv, b)

    return out


# Just DEConvolution 2d with it's weights and biases
def deconv2d(name, x, output_shape, kernel_size=(3, 3), padding='SAME', stride=(1, 1)):
    """
    Deconv2d wrapper
    :param name:
    :param x:
    :param output_shape:
    :param kernel_size:
    :param padding:
    :param stride:
    :param initializer:
    :return:
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], output_shape[-1], x.get_shape()[-1]]

        w = get_deconv_filter(kernel_shape)
        deconv = tf.nn.conv2d_transpose(x, w, tf.stack(output_shape), strides=stride, padding=padding)

        b = tf.get_variable('biases_deconv', [output_shape[-1]], initializer=tf.constant_initializer(0.01))
        out = tf.nn.bias_add(deconv, b)

    return out


class ConvLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """A LSTM cell with convolutions instead of multiplications.
    Reference:
      Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
    """

    def __init__(self,
                 scope,
                 w,
                 b,
                 num_units,
                 padding='SAME',
                 stride=(1, 1),
                 normalize=False,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tf.tanh,
                 reuse=None,
                 ):
        self._scope = scope
        self._w = w
        self._b = b
        self._stride = [stride[0], stride[1]]
        self._padding = padding
        self._normalize = normalize
        tf.nn.rnn_cell.BasicLSTMCell.__init__(self, num_units, forget_bias, state_is_tuple, activation, reuse)

    def call(self, inputs, state):

        with tf.variable_scope(self._scope or self.__class__.__name__):
            previous_output, previous_memory = state

            inputs = tf.concat([inputs, previous_output], axis=3)

            y = tf.nn.convolution(inputs, self._w, padding=self._padding, strides=self._stride)

            if not self._normalize:
                y += self._b

            input_contribution, input_gate, forget_gate, output_gate = tf.split(y, 4, axis=3)

            if self._normalize:
                input_contribution = tf.contrib.layers.layer_norm(input_contribution)
                input_gate = tf.contrib.layers.layer_norm(input_gate)
                forget_gate = tf.contrib.layers.layer_norm(forget_gate)
                output_gate = tf.contrib.layers.layer_norm(output_gate)

            memory = (previous_memory
                      * tf.sigmoid(forget_gate + self._forget_bias)
                      + tf.sigmoid(input_gate) * self._activation(input_contribution))

            if self._normalize:
                memory = tf.contrib.layers.layer_norm(memory)

            output = self._activation(memory) * tf.sigmoid(output_gate)

        return output, tf.contrib.rnn.LSTMStateTuple(output, memory)


def convlstmtst(scope, x,
                input_shape,
                filters,
                kernel=(3, 3),
                padding='SAME',
                stride=(1, 1),
                initializer=tf.contrib.layers.xavier_initializer(),
                normalize=True,
                forget_bias=1.0,
                state_is_tuple=True,
                activation=tf.tanh,
                reuse=None, ):
    with tf.variable_scope(scope) as scope_:
        # Initialize the weights and the biases
        shape = [input_shape[1], input_shape[2]]
        channels = input_shape[-1]
        gates = 4 * filters if filters > 1 else 4
        n = channels + filters
        m = gates
        w = tf.get_variable('w_conv_lstm', list(kernel) + [n, m], initializer=initializer)
        variable_summaries(w)
        b = tf.get_variable('b_conv_lstm', [m], initializer=tf.constant_initializer(0.0))
        variable_summaries(b)
        # init the num of units
        num_units = tf.TensorShape(shape + [filters])

        # init the cell
        cell = ConvLSTMCell(scope_, w, b, num_units,
                            padding=padding, stride=stride, normalize=normalize,
                            forget_bias=forget_bias, state_is_tuple=state_is_tuple, activation=activation, reuse=reuse)

        state = get_state_variables(input_shape[0], cell)

        # call the cell
        output, new_state = cell.call(x, state)

        # clearing the state
        clear_op = get_clear_state_op(state, input_shape[0], cell)
        # update the init_state
        update_op = get_state_update_op(state, new_state)

    return output, update_op, clear_op


def get_state_variables(batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_c, state_h = cell.zero_state(batch_size, tf.float32)
    state = tf.nn.rnn_cell.LSTMStateTuple(
        tf.Variable(state_c, trainable=False, name='state_c'),
        tf.Variable(state_h, trainable=False, name='state_h'))
    # Return as a tuple
    return state


def get_clear_state_op(state_variable, batch_size, cell):
    update_ops = []
    state_c, state_h = cell.zero_state(batch_size, tf.float32)
    update_ops.extend([state_variable[0].assign(state_c),
                       state_variable[1].assign(state_h)])
    return update_ops


def get_state_update_op(state_variable, new_state):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    update_ops.extend([state_variable[0].assign(new_state[0]),
                       state_variable[1].assign(new_state[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return update_ops


# Just added some summaries and init the weights and it's biases
def convlstm(scope, x, input_shape, filters, kernel=(3, 3), padding='SAME', stride=(1, 1),
             initializer=tf.contrib.layers.xavier_initializer(), forget_bias=1.0, activation=tf.tanh,
             normalize=True):
    with tf.variable_scope(scope) as scope_:
        # Initialize the weights
        shape = [input_shape[2], input_shape[3]]
        channels = input_shape[-1]
        gates = 4 * filters if filters > 1 else 4
        n = channels + filters
        m = gates
        w = tf.get_variable('w_conv_lstm', list(kernel) + [n, m], initializer=initializer)
        variable_summaries(w)
        b = tf.get_variable('b_conv_lstm', [m], initializer=tf.constant_initializer(0.0))
        variable_summaries(b)

        # Prepare the Input
        # x = tf.reshape(x, tf.stack(input_shape))
        input_lstm_transposed = tf.transpose(x, (1, 0, 2, 3, 4))

        # init the cell
        cell = ConvLSTMCell("conv_lstm_cell", w, b, shape, filters, kernel, padding, stride, initializer, forget_bias,
                            activation, normalize)

        # init the rnn
        outputs, state = tf.nn.dynamic_rnn(cell, input_lstm_transposed, dtype=input_lstm_transposed.dtype,
                                           time_major=True)

        # Prepare the output
        outputs = tf.transpose(outputs, (1, 0, 2, 3, 4))

    return outputs, state


def convlstm_cell(scope, x, h, input_shape, filters, kernel=(3, 3), padding='SAME', stride=(1, 1),
                  initializer=tf.contrib.layers.xavier_initializer(),
                  forget_bias=1.0,
                  activation=tf.tanh,
                  normalize=True):
    with tf.variable_scope(scope) as scope_:
        # Initialize the weights
        shape = [input_shape[1], input_shape[2]]
        channels = input_shape[-1]
        gates = 4 * filters if filters > 1 else 4
        n = channels + filters
        m = gates
        w = tf.get_variable('w_conv_lstm', list(kernel) + [n, m], initializer=initializer)
        variable_summaries(w)
        b = tf.get_variable('b_conv_lstm', [m], initializer=tf.constant_initializer(0.0))
        variable_summaries(b)

        # init the cell
        cell = ConvLSTMCell("conv_lstm_cell", w, b, shape, filters, kernel, padding, stride, initializer, forget_bias,
                            activation, normalize)

        # call the cell
        if h is None:
            h = cell.zero_state(tf.shape(x)[0], tf.float32)

        output, state = cell(x, h)

    return output, state


def max_pool_2d_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2d_2x2_masks(x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def max_unpool_2d_2x2(scope, x, ind, ksize=(1, 2, 2, 1)):
    """
    Unpooling layer after max_pool_with_argmax.
    :param scope:
    :param x: max pooled output tensor
    :param ind: argmax indices
    :param ksize: ksize is the same as for the pool
    :return: unpooling tensor
    """

    with tf.variable_scope(scope):
        input_shape = x.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
        pool_ = tf.reshape(x, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
        ind_ = tf.reshape(ind, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
        ind_ = tf.concat(axis=1, values=[b, ind_])
        ref = tf.Variable(tf.zeros([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]))
        ret = tf.scatter_nd_update(ref, ind_, pool_)
        ret = tf.reshape(ret, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
        return ret


def max_pool_3d_2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME')


def conv2d_adv(name, x, output_dim, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=-1, trainable=True):
    """
    Advanced Conv2d Wrapper
    :param name:
    :param x:
    :param output_dim:
    :param kernel_size:
    :param padding:
    :param stride:
    :param initializer:
    :param l2_strength:(weight decay)
    :param bias:
    :param trainable: trainable variables
    :return out, weights, biases:
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        with tf.variable_scope('weights'):
            w = _variable_with_weight_decay(kernel_shape, initializer, l2_strength, trainable)
            variable_summaries(w)
        if bias != -1:
            with tf.variable_scope('biases'):
                b = tf.get_variable('biases_conv', [output_dim], initializer=tf.constant_initializer(bias), trainable=trainable)
                variable_summaries(b)
        with tf.variable_scope('conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            if bias == -1:
                out = conv
            else:
                out = tf.nn.bias_add(conv, b)

    return out


def atrous_conv2d_adv(name, x, output_dim, kernel_size=(3, 3), rate=2, padding='SAME',
                      initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
        Advanced Dilated Conv2d Wrapper
        :param name:
        :param x:
        :param output_dim:
        :param kernel_size:
        :param rate:
        :param padding:
        :param initializer:
        :param l2_strength:(weight decay)
        :param bias:
        :return out, weights, biases:
        """
    with tf.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        with tf.variable_scope('weights'):
            w = _variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.variable_scope('biases'):
            b = tf.get_variable('biases_conv', [output_dim], initializer=tf.constant_initializer(bias))
            variable_summaries(b)
        with tf.variable_scope('atrous_conv2d'):
            conv = tf.nn.atrous_conv2d(x, w, rate, padding)
            out = tf.nn.bias_add(conv, b)

    return out


def _variable_with_weight_decay(kernel_shape, initializer, wd, trainable=True):
    w = tf.get_variable('w_conv', kernel_shape, tf.float32, initializer=initializer, trainable=trainable)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(
            tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    variable_summaries(w)
    return w


def conv2d_x2_pool(name, x, output_dim, kernel_size=(3, 3),
                   padding='SAME', stride=(1, 1),
                   initializer=tf.contrib.layers.xavier_initializer(),
                   l2_strength=2e-4, bias=-1, use_batchnorm=False,
                   is_training=True, pooling=True, activation=tf.nn.relu, trainable=True):
    with tf.variable_scope(name + '_conv1') as scope1:
        conv1 = conv2d_adv(scope1, x, output_dim[0], kernel_size=kernel_size, stride=stride,
                           l2_strength=l2_strength,
                           bias=bias, initializer=initializer, padding=padding, trainable=trainable)
        if use_batchnorm:
            conv1_bn = tf.layers.batch_normalization(conv1, training=is_training, trainable=trainable)
            conv1_o = activation(conv1_bn)
        else:
            conv1_o = activation(conv1)

    with tf.variable_scope(name + '_conv2') as scope1:
        conv2 = conv2d_adv(scope1, conv1_o, output_dim[1], kernel_size=kernel_size, stride=stride,
                           l2_strength=l2_strength,
                           bias=bias, initializer=initializer, padding=padding, trainable=trainable)
        if use_batchnorm:
            conv2_bn = tf.layers.batch_normalization(conv2, training=is_training, trainable=trainable)
            conv2_o = activation(conv2_bn)
        else:
            conv2_o = activation(conv2)

        if pooling:
            pool = max_pool_2d_2x2(conv2_o)
            return conv2_o, pool
        return conv2_o


def upsampling_2d(tensor, size=(2, 2)):
    """
    Args:
    Returns:
    """
    H, W, _ = tensor.get_shape().as_list()[1:]

    mul_h, mul_w = size
    output_H = H * mul_h
    output_W = W * mul_w
    return tf.image.resize_bilinear(tensor, (output_H, output_W), align_corners=None)


def upsampling_2d_concat(input_a, input_b, name):
    """Upsample input_a and concat with input_b
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
    Returns:
    """
    with tf.variable_scope(name) as scope:
        upsample = upsampling_2d(input_a, size=(2, 2))
        return tf.concat([upsample, input_b], axis=-1)
