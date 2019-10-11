import tensorflow as tf
from Training.RGB.base_model import BaseModel
import math


class SingleViewModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if self.config.p_stacking_frames and self.config.f_stacking_frames:
            self.X = tf.placeholder(tf.float32,
                                    [None, self.config.img_h, self.config.img_w,
                                     self.config.p_stacking_frames+self.config.f_stacking_frames], 'X')
        elif self.config.p_stacking_frames:
            self.X = tf.placeholder(tf.float32,
                                    [None, self.config.img_h, self.config.img_w, self.config.p_stacking_frames], 'X')
        else:
            self.X = tf.placeholder(tf.float32, [None, self.config.img_h, self.config.img_w, 3], 'X')
        if self.config.speed_input:
            self.speed_x = tf.placeholder(tf.float32, [None, 1], 'forward_speed')
        if self.config.separate_throttle_brake:
            self.y = tf.placeholder(tf.float32, [None, 3], 'y')
        else:
            self.y = tf.placeholder(tf.float32, [None, 2], 'y')
        self.training = tf.placeholder(tf.bool)
        self.build_model()
        self.init_saver()

    def dropout_with_keep(self, input):
        return tf.nn.dropout(input, 0.5)

    def dropout_no_keep(self, input):
        return tf.nn.dropout(input, 1.0)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):
        if self.config.normalized_input:
            self.X = self.X / 255.0
        conv1 = tf.layers.conv2d(self.X, 32, 5, strides=2, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='VALID')
        if self.config.batch_norm:
            conv1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=self.training)
        conv1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(conv1, 32, 3, strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='VALID')
        if self.config.batch_norm:
            conv2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=self.training)
        conv2 = tf.nn.relu(conv2)

        conv3 = tf.layers.conv2d(conv2, 64, 3, strides=2, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='VALID')
        if self.config.batch_norm:
            conv3 = tf.contrib.layers.batch_norm(conv3, center=True, scale=True, is_training=self.training)
        conv3 = tf.nn.relu(conv3)

        conv4 = tf.layers.conv2d(conv3, 64, 3, strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='VALID')
        if self.config.batch_norm:
            conv4 = tf.contrib.layers.batch_norm(conv4, center=True, scale=True, is_training=self.training)
        conv4 = tf.nn.relu(conv4)

        conv5 = tf.layers.conv2d(conv4, 128, 3, strides=2, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='VALID')
        if self.config.batch_norm:
            conv5 = tf.contrib.layers.batch_norm(conv5, center=True, scale=True, is_training=self.training)
        conv5 = tf.nn.relu(conv5)

        conv6 = tf.layers.conv2d(conv5, 128, 3, strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='VALID')
        if self.config.batch_norm:
            conv6 = tf.contrib.layers.batch_norm(conv6, center=True, scale=True, is_training=self.training)
        conv6 = tf.nn.relu(conv6)
        """
        conv7 = tf.layers.conv2d(conv6, 256, 3, strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='VALID')
        if self.config.batch_norm:
            conv7 = tf.contrib.layers.batch_norm(conv7, center=True, scale=True, is_training=self.training)
        conv7 = tf.nn.relu(conv7)

        conv8 = tf.layers.conv2d(conv7, 256, 3, strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 padding='VALID')
        if self.config.batch_norm:
            conv8 = tf.contrib.layers.batch_norm(conv8, center=True, scale=True, is_training=self.training)
        conv8 = tf.nn.relu(conv8)
        """
        conv8 = conv6
        # Flattening
        self.flattened_layer = tf.contrib.layers.flatten(conv8)
        fc1 = tf.layers.dense(inputs=self.flattened_layer, units=256,
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        if self.config.dropout:
            fc1 = tf.layers.dropout(fc1, rate=0.5, training=self.training)
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.layers.dense(inputs=fc1, units=256, kernel_initializer=tf.contrib.layers.xavier_initializer())
        if self.config.dropout:
            fc2 = tf.layers.dropout(fc2, rate=0.5, training=self.training)
        fc2 = tf.nn.relu(fc2)
        # Add speed path to the network
        if self.config.speed_input:
            fc1_speed = tf.layers.dense(inputs=self.speed_x, units=128,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            if self.config.dropout:
                fc1_speed = tf.layers.dropout(fc1_speed, rate=0.5, training=self.training)
            fc1_speed = tf.nn.relu(fc1_speed)
            fc2_speed = tf.layers.dense(inputs=fc1_speed, units=128,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            if self.config.dropout:
                fc2_speed = tf.layers.dropout(fc2_speed, rate=0.5, training=self.training)
            fc2_speed = tf.nn.relu(fc2_speed)
            concat = tf.concat([fc2, fc2_speed], 1)
            fc_concat = tf.layers.dense(inputs=concat, units=512,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            if self.config.dropout:
                fc_concat = tf.layers.dropout(fc_concat, rate=0.5, training=self.training)
            fc_concat = tf.nn.relu(fc_concat)
        else:
            fc_concat = fc2

        # Branching on 4 heads
        # Head 0
        fc1_head_0 = tf.layers.dense(inputs=fc_concat, units=256,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        if self.config.dropout:
            fc1_head_0 = tf.layers.dropout(fc1_head_0, rate=0.5, training=self.training)
        fc1_head_0 = tf.nn.relu(fc1_head_0)
        fc2_head_0 = tf.layers.dense(inputs=fc1_head_0, units=256,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        if self.config.dropout:
            fc2_head_0 = tf.layers.dropout(fc2_head_0, rate=0.5, training=self.training)
        fc2_head_0 = tf.nn.relu(fc2_head_0)
        if self.config.separate_throttle_brake:
            self.output_0 = tf.layers.dense(inputs=fc2_head_0, units=3,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        else:
            self.output_0 = tf.layers.dense(inputs=fc2_head_0, units=2,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # Head 1
        fc1_head_1 = tf.layers.dense(inputs=fc_concat, units=256,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        if self.config.dropout:
            fc1_head_1 = tf.layers.dropout(fc1_head_1, rate=0.5, training=self.training)
        fc1_head_1 = tf.nn.relu(fc1_head_1)
        fc2_head_1 = tf.layers.dense(inputs=fc1_head_1, units=256,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        if self.config.dropout:
            fc2_head_1 = tf.layers.dropout(fc2_head_1, rate=0.5, training=self.training)
        fc2_head_1 = tf.nn.relu(fc2_head_1)
        if self.config.separate_throttle_brake:
            self.output_1 = tf.layers.dense(inputs=fc2_head_1, units=3,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        else:
            self.output_1 = tf.layers.dense(inputs=fc2_head_1, units=2,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # Head 2
        fc1_head_2 = tf.layers.dense(inputs=fc_concat, units=256,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        if self.config.dropout:
            fc1_head_2 = tf.layers.dropout(fc1_head_2, rate=0.5, training=self.training)
        fc1_head_2 = tf.nn.relu(fc1_head_2)
        fc2_head_2 = tf.layers.dense(inputs=fc1_head_2, units=256,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        if self.config.dropout:
            fc2_head_2 = tf.layers.dropout(fc2_head_2, rate=0.5, training=self.training)
        fc2_head_2 = tf.nn.relu(fc2_head_2)
        if self.config.separate_throttle_brake:
            self.output_2 = tf.layers.dense(inputs=fc2_head_2, units=3,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        else:
            self.output_2 = tf.layers.dense(inputs=fc2_head_2, units=2,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # Head 3
        fc1_head_3 = tf.layers.dense(inputs=fc_concat, units=256,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        if self.config.dropout:
            fc1_head_3 = tf.layers.dropout(fc1_head_3, rate=0.5, training=self.training)
        fc1_head_3 = tf.nn.relu(fc1_head_3)
        fc2_head_3 = tf.layers.dense(inputs=fc1_head_3, units=256,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        if self.config.dropout:
            fc2_head_3 = tf.layers.dropout(fc2_head_3, rate=0.5, training=self.training)
        fc2_head_3 = tf.nn.relu(fc2_head_3)
        if self.config.separate_throttle_brake:
            self.output_3 = tf.layers.dense(inputs=fc2_head_3, units=3,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        else:
            self.output_3 = tf.layers.dense(inputs=fc2_head_3, units=2,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.contrast_node = tf.image.random_contrast(self.X, lower=self.config.auggmentation_contrast_lower,
                                                      upper=self.config.auggmentation_contrast_upper)
        self.brightness_node = tf.image.random_brightness(self.X,
                                                          max_delta=self.config.auggmentation_brightness_max_delta)

        if self.config.separate_throttle_brake:
            self.loss_0_a = tf.losses.mean_squared_error(self.y[:, 0], self.output_0[:, 0])
            self.loss_0_s = tf.losses.mean_squared_error(self.y[:, 1], self.output_0[:, 1])
            self.loss_0_b = tf.losses.mean_squared_error(self.y[:, 2], self.output_0[:, 2])
            self.loss_0 = tf.reduce_mean(self.loss_0_a + self.loss_0_s + self.loss_0_b)
        else:
            self.loss_0_a = tf.losses.mean_squared_error(self.y[:, 0], self.output_0[:, 0])
            self.loss_0_s = tf.losses.mean_squared_error(self.y[:, 1], self.output_0[:, 1])
            self.loss_0 = tf.reduce_mean(self.loss_0_s + self.loss_0_a)

        if self.config.separate_throttle_brake:
            self.loss_1_a = tf.losses.mean_squared_error(self.y[:, 0], self.output_1[:, 0])
            self.loss_1_s = tf.losses.mean_squared_error(self.y[:, 1], self.output_1[:, 1])
            self.loss_1_b = tf.losses.mean_squared_error(self.y[:, 2], self.output_1[:, 2])
            self.loss_1 = tf.reduce_mean(self.loss_1_a + self.loss_1_s + self.loss_1_b)
        else:
            self.loss_1_a = tf.losses.mean_squared_error(self.y[:, 0], self.output_1[:, 0])
            self.loss_1_s = tf.losses.mean_squared_error(self.y[:, 1], self.output_1[:, 1])
            self.loss_1 = tf.reduce_mean(self.loss_1_s + self.loss_1_a)

        if self.config.separate_throttle_brake:
            self.loss_2_a = tf.losses.mean_squared_error(self.y[:, 0], self.output_2[:, 0])
            self.loss_2_s = tf.losses.mean_squared_error(self.y[:, 1], self.output_2[:, 1])
            self.loss_2_b = tf.losses.mean_squared_error(self.y[:, 2], self.output_2[:, 2])
            self.loss_2 = tf.reduce_mean(self.loss_2_a + self.loss_2_s + self.loss_2_b)
        else:
            self.loss_2_a = tf.losses.mean_squared_error(self.y[:, 0], self.output_2[:, 0])
            self.loss_2_s = tf.losses.mean_squared_error(self.y[:, 1], self.output_2[:, 1])
            self.loss_2 = tf.reduce_mean(self.loss_2_s + self.loss_2_a)

        if self.config.separate_throttle_brake:
            self.loss_3_a = tf.losses.mean_squared_error(self.y[:, 0], self.output_3[:, 0])
            self.loss_3_s = tf.losses.mean_squared_error(self.y[:, 1], self.output_3[:, 1])
            self.loss_3_b = tf.losses.mean_squared_error(self.y[:, 2], self.output_3[:, 2])
            self.loss_3 = tf.reduce_mean(self.loss_3_a + self.loss_3_s + self.loss_3_b)
        else:
            self.loss_3_a = tf.losses.mean_squared_error(self.y[:, 0], self.output_3[:, 0])
            self.loss_3_s = tf.losses.mean_squared_error(self.y[:, 1], self.output_3[:, 1])
            self.loss_3 = tf.reduce_mean(self.loss_3_s + self.loss_3_a)

        # Add decayed learning rate:
        if self.config.decay_lr:
            self.decayed_lr = tf.train.exponential_decay(learning_rate=self.config.learning_rate, decay_steps=2700,
                                                         global_step=self.global_step_tensor, decay_rate=0.96,
                                                         staircase=False)
            optimizer = tf.train.AdamOptimizer(self.decayed_lr)
        else:
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op_0 = optimizer.minimize(self.loss_0, global_step=self.global_step_tensor)
            self.train_op_1 = optimizer.minimize(self.loss_1, global_step=self.global_step_tensor)
            self.train_op_2 = optimizer.minimize(self.loss_2, global_step=self.global_step_tensor)
            self.train_op_3 = optimizer.minimize(self.loss_3, global_step=self.global_step_tensor)

        self.v_grads_s = tf.gradients(self.output_0, self.X)[0]
        self.v_grads_s_abs = tf.abs(self.v_grads_s)  # tf.image.adjust_contrast(tf.abs(self.v_grads_s), 0.85)

        if self.config.apply_auggmentation:
            self.contrast_node = tf.image.random_contrast(self.X, lower=self.config.auggmentation_contrast_lower,
                                                          upper=self.config.auggmentation_contrast_upper)
            self.brightness_node = tf.image.random_brightness(self.X,
                                                              max_delta=self.config.auggmentation_brightness_max_delta)
