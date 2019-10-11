# -*- coding: utf-8 -*-
# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# @author: german,felipecode


from __future__ import print_function
import abc
from ..carla.client import VehicleControl
from driving_benchmarks.testing_config import Config
import sys
import numpy as np
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import math
import copy
import tensorflow as tf
from Training.RGB.single_view_model import SingleViewModel
from Training.RGB.utils.model_summary import print_network_state



class Agent(object):
    def __init__(self):
        self.__metaclass__ = abc.ABCMeta
        self.t_config = Config()
        self.old_imgs_count = 0
        self.push_counter = 0
        self.old_imgs = []
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.model = SingleViewModel(self.t_config)
        self.model.load(self.sess, self.t_config.load)
        print_network_state()

    def restart(self):
        print("restarting agent")
        self.old_imgs_count = 0
        self.push_counter = 0
        self.old_imgs = []

    @abc.abstractmethod
    def run_step(self, measurements, sensor_data, directions, target):
        """
        Function to be redefined by an agent.
        :param The measurements like speed, the image data and a target
        :returns A carla Control object, with the steering/gas/brake for the agent
        """

class Stacking_previous_Agent(Agent):
    """
    Simple derivation of Agent Class,
    A trivial agent agent that goes straight
    """
    def run_step(self, measurements, sensor_data, directions, target):
        # Input image: sensor_data['CameraRGB'].data
        # Input speed: measurements.player_measurements.forward_speed
        # direction: directions-2
        forward_img = sensor_data['CameraRGB'].data
        forward_img = cv2.resize(forward_img, (200, 200))
        direction = directions - 2
        #print("direction = ", direction)
        input_img = forward_img[self.t_config.clip_until:, :, :]

        if self.t_config.p_stacking_frames:
            temp_gray = []
            for k in range(self.old_imgs_count):
                # add old images
                temp_gray.append(self.old_imgs[k])
            for k in range(self.t_config.p_stacking_frames - 1 - self.old_imgs_count):
                # if not enough old images repeat the last image
                # will enter here first time only
                temp_gray.append(cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY))
                # execute in the last iteration in the loop
                if k == self.t_config.p_stacking_frames - 2 - self.old_imgs_count:
                    self.old_imgs = copy.deepcopy(temp_gray)
                    self.old_imgs_count = self.t_config.p_stacking_frames - 1
            temp_gray.append(cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY))
            # update self.old_imgs
            for k in range(self.t_config.p_stacking_frames - 1):
                if k == self.t_config.p_stacking_frames - 2:
                    continue
                else:
                    self.old_imgs[k] = self.old_imgs[k + 1]
            self.old_imgs[-1] = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img = np.swapaxes(np.swapaxes(np.asarray(temp_gray), 0, 1), 1, 2)

        input_img = np.expand_dims(input_img, axis=0)

        if self.t_config.separate_throttle_brake:
            y = np.zeros((1, 3))
        else:
            y = np.zeros((1, 2))

        forward_speed = np.zeros((1, 1))
        forward_speed[0] = measurements.player_measurements.forward_speed * 3.6
        # Run my model
        if self.t_config.speed_input:
            feed_dict = {self.model.X: input_img, self.model.speed_x: forward_speed,
                         self.model.y: y, self.model.training: self.t_config.is_training}
        else:
            feed_dict = {self.model.X: input_img, self.model.y: y, self.model.training: self.t_config.is_training}
        # Choose the correct branch
        if direction == 0:
            model_output = self.sess.run([self.model.output_0], feed_dict=feed_dict)
        elif direction == 1:
            model_output = self.sess.run([self.model.output_1], feed_dict=feed_dict)
        elif direction == 2:
            model_output = self.sess.run([self.model.output_2], feed_dict=feed_dict)
        elif direction == 3:
            model_output = self.sess.run([self.model.output_3], feed_dict=feed_dict)
        else:
            model_output = self.sess.run([self.model.output_0], feed_dict=feed_dict)
            print("Invalid direction = ", direction)
            print("As direction should be in range (from 0 to 3)")

        # for debugging purpose
        #print("Throttle = ", model_output[0][0][0])
        #print("Brake = ", model_output[0][0][2])


        if self.t_config.separate_throttle_brake:
            if model_output[0][0][1] > 1:
                model_output[0][0][1] = 1.0
            elif model_output[0][0][1] < -1:
                model_output[0][0][1] = -1.0
            if model_output[0][0][0] > 1:
                model_output[0][0][0] = 1.0
            elif model_output[0][0][0] < 0:
                model_output[0][0][0] = 0.0
            if model_output[0][0][2] > 1:
                model_output[0][0][2] = 1.0
            elif model_output[0][0][2] < 0:
                model_output[0][0][2] = 0.0
        else:
            if model_output[0][0][1] > 1:
                model_output[0][0][1] = 1.0
            elif model_output[0][0][1] < -1:
                model_output[0][0][1] = -1.0
            if model_output[0][0][0] > 1:
                model_output[0][0][0] = 1.0
            elif model_output[0][0][0] < 0:
                model_output[0][0][0] = 0.0

        control = VehicleControl()
        if self.t_config.give_car_push and self.push_counter < 30:
            self.push_counter += 1
            control.throttle = 0.5
            control.steer = model_output[0][0][1]
            control.brake = 0
        else:
            control.throttle = model_output[0][0][0]
            control.steer = model_output[0][0][1]
            control.brake = model_output[0][0][2]
        control.hand_brake = False

        return control
