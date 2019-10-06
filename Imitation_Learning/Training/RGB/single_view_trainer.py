import tensorflow as tf
from Training.RGB.base_train import BaseTrain
from tqdm import tqdm

import numpy as np
import math
from PIL import ImageFilter
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import random
from matplotlib import pyplot
from copy import deepcopy
from scipy import misc
import matplotlib.pyplot as plt


class Trainer(BaseTrain):
    def __init__(self, sess, model, data_loader, val_loader, config):
        super().__init__(sess, model, None, config)
        self.model = model
        self.config = config
        self.sess = sess
        self.data_loader = data_loader
        self.val_loader = val_loader

    def train(self):
        ag_cnt = 0
        for epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.epochs):
            # print the current learning rate to check the decaying process
            lr = self.sess.run([self.model.decayed_lr])
            print("the cuurent learning rate = ", lr)
            losses = []
            losses_val = []
            losses_s_a = []
            losses_s_s = []
            losses_s_b = []
            losses_s_a_val = []
            losses_s_s_val = []
            losses_s_b_val = []
            for itr in tqdm(range(self.data_loader.num_of_episodes)):
                measurement = np.load(
                    self.data_loader.config.labels_dir + '/' + self.data_loader.measurements_files[itr])
                # Generate list of random numbers without repetition
                item_num = random.sample(range(len(measurement)), len(measurement))
                for i in range(int(np.ceil(len(measurement) / self.config.batch_size))):
                    stacked_batch = []
                    start = self.config.batch_size * i
                    end = start + self.config.batch_size
                    if end >= len(measurement):
                        end = len(measurement)
                        start = end - self.config.batch_size
                    if start < 0:
                        start = 0
                    x_s, y = self.data_loader.get_batch(episode_num=itr, item_nums=item_num[start:end])

                    if self.config.apply_auggmentation:
                        # Apply noise on a portion of the data.
                        rand = random.randint(1, self.config.apply_auggmentation_every)
                        if rand == 1:
                            auggmented_batch = self.augment(x_s, self.sess)
                            x_s = auggmented_batch.copy()
                            ag_cnt += 3

                    if self.config.stacking_frames:
                        temp = []
                        temp_gray = []
                        for num in range(len(x_s)):
                            temp.append(x_s[num])
                            if (num + 1) % self.config.stacking_frames == 0:
                                for k in range(self.config.stacking_frames):
                                    temp_gray.append(cv2.cvtColor(temp[k], cv2.COLOR_BGR2GRAY))
                                stacked_batch.append(
                                    np.swapaxes(np.swapaxes(np.asarray(temp_gray), 0, 1), 1, 2))
                                temp = []
                                temp_gray = []
                        input_images = stacked_batch
                    else:
                        input_images = x_s
                    if self.config.speed_input:
                        feed_dict = {self.model.X: input_images,
                                     self.model.speed_x: np.expand_dims(y[:, 3], axis=-1),
                                     self.model.y: y[:, :self.config.num_of_Actions], self.model.training: True}
                    else:
                        feed_dict = {self.model.X: input_images, self.model.y: y[:, :self.config.num_of_Actions],
                                     self.model.training: True}

                    if self.config.separate_throttle_brake:
                        _, loss_s_a, loss_s_s, loss_s_b = self.sess.run([self.model.train_op_s, self.model.loss_s_a,
                                                                         self.model.loss_s_s, self.model.loss_s_b],
                                                                        feed_dict=feed_dict)
                        losses_s_a.append(loss_s_a)
                        losses_s_s.append(loss_s_s)
                        losses_s_b.append(loss_s_b)
                        losses.append(loss_s_a)
                        losses.append(loss_s_s)
                        losses.append(loss_s_b)
                    else:
                        _, loss_s_a, loss_s_s = self.sess.run(
                            [self.model.train_op_s, self.model.loss_s_a, self.model.loss_s_s], feed_dict=feed_dict)
                        losses_s_a.append(loss_s_a)
                        losses_s_s.append(loss_s_s)
                        losses.append(loss_s_a)
                        losses.append(loss_s_s)

            summaries_dict = {"straight_loss_steer": np.mean(losses_s_s),
                              "straight_loss_acceleration": np.mean(losses_s_a),
                              "total_loss": np.mean(losses)}

            self.summarize(self.model.cur_epoch_tensor.eval(self.sess), summaries_dict=summaries_dict)
            self.model.save(self.sess)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            print('epoch: {0}\tloss: {1}'.format(epoch, np.mean(losses)))
            # print('agmented frame count: ', ag_cnt)
            ########################################################################################################
            #                                            Validation                                                #
            ########################################################################################################
            for itr in tqdm(range(self.val_loader.num_of_episodes)):
                measurement = np.load(
                    self.val_loader.config.labels_dir + '/' + self.val_loader.measurements_files[itr])
                # Generate list of random numbers without repetition
                item_num = random.sample(range(len(measurement)), len(measurement))
                for i in range(int(np.ceil(len(measurement) / self.config.batch_size))):
                    stacked_batch = []
                    start = self.config.batch_size * i
                    end = start + self.config.batch_size
                    if end >= len(measurement):
                        end = len(measurement)
                        start = end - self.config.batch_size
                    if start < 0:
                        start = 0

                    x_s, y = self.val_loader.get_batch(episode_num=itr, item_nums=item_num[start:end])

                    if self.config.stacking_frames:
                        temp = []
                        temp_gray = []
                        for num in range(len(x_s)):
                            temp.append(x_s[num])
                            if (num + 1) % self.config.stacking_frames == 0:
                                for k in range(self.config.stacking_frames):
                                    temp_gray.append(cv2.cvtColor(temp[k], cv2.COLOR_BGR2GRAY))
                                stacked_batch.append(
                                    np.swapaxes(np.swapaxes(np.asarray(temp_gray), 0, 1), 1, 2))
                                temp = []
                                temp_gray = []
                        input_images = stacked_batch
                    else:
                        input_images = x_s

                    if self.config.speed_input:
                        feed_dict = {self.model.X: input_images,
                                     self.model.speed_x: np.expand_dims(y[:, 3], axis=-1),
                                     self.model.y: y[:, :self.config.num_of_Actions], self.model.training: False}
                    else:
                        feed_dict = {self.model.X: input_images, self.model.y: y[:, :self.config.num_of_Actions],
                                     self.model.training: False}

                    if self.config.separate_throttle_brake:
                        loss_s_a_val, loss_s_s_val, loss_s_b_val = self.sess.run(
                            [self.model.loss_s_a, self.model.loss_s_s, self.model.loss_s_b], feed_dict=feed_dict)
                        losses_s_a_val.append(loss_s_a_val)
                        losses_s_s_val.append(loss_s_s_val)
                        losses_s_b_val.append(loss_s_b_val)
                        losses_val.append(loss_s_a_val)
                        losses_val.append(loss_s_s_val)
                        losses_val.append(loss_s_b_val)
                    else:
                        loss_s_a_val, loss_s_s_val = self.sess.run(
                            [self.model.loss_s_a, self.model.loss_s_s], feed_dict=feed_dict)
                        losses_s_a_val.append(loss_s_a_val)
                        losses_s_s_val.append(loss_s_s_val)
                        losses_val.append(loss_s_a_val)
                        losses_val.append(loss_s_s_val)

            print('epoch: {0}\tVal_loss: {1}'.format(epoch, np.mean(losses_val)))
            # write loss info in text:
            f = open(self.config.loss_filename, "a+")
            f.write('epoch: {0}\tloss: {1}'.format(epoch, np.mean(losses)))
            f.write('epoch: {0}\tVal_loss: {1}'.format(epoch, np.mean(losses_val)))
        f.close()

    def augment(self, batch, sess=None):
        #res = deepcopy(batch[:, :, :, :])
        res = deepcopy(np.asarray(batch)[:, :, :, :])
        auggmentation_count = random.randint(1, self.config.auggmentation_counts)
        auggmentation_set = random.sample(range(1, auggmentation_count + 1), auggmentation_count)
        if (self.config.auggmentation_blur and (1 in auggmentation_set)):
            # print('Applying blurring...')
            res = self.add_blur(res)
        if (self.config.auggmentation_contrast and (2 in auggmentation_set)):
            # print('Applying contrast...')
            res = self.apply_contrast(res, sess)
        if (self.config.auggmentation_brightness and (3 in auggmentation_set)):
            # print('Applying brightness...')
            res = self.apply_brightness(res, sess)
        if (self.config.auggmentation_noise_salt_and_pepper and (4 in auggmentation_set)):
            # print('Applying salt and pepper noise...')
            res = self.add_salt_pepper_noise(res)
        return res

    def save_image(self, X_imgs, type, agmentation_array=None, tracking=0):
        if type == 'aug':
            print('X_imgs augmentation for saving: ', X_imgs.shape)
            cv2.imwrite('/media/user/noname/loay/carla_data/data/augmentation/unit_test-samples/augmented-' + str(
                tracking) + '-' + ''.join(str(e) for e in agmentation_array) + '.png', np.squeeze(X_imgs))
            # pyplot.imsave("/media/user/noname/loay/carla_data/data/augmentation/unit_test-samples/augmented.png",np.squeeze(X_imgs[i]))
        else:
            print('X_imgs original for saving: ', X_imgs.shape)
            cv2.imwrite(
                '/media/user/noname/loay/carla_data/data/augmentation/unit_test-samples/original-' + str(
                    tracking) + '.png',
                np.squeeze(X_imgs))
            # X_imgs /=255.0
            # pyplot.imsave("/media/user/noname/loay/carla_data/data/augmentation/unit_test-samples/original.png",
            #               np.squeeze(X_imgs[i]))

    def apply_brightness(self, X_imgs, sess=None):
        # print('input_batch_shape: ',X_imgs.shape)
        result_batch = deepcopy(X_imgs)
        # print('result_batch_shape: ',result_batch.shape)
        for i in range(X_imgs.shape[0]):
            # print('image number (',i,')s shape is: ',X_imgs[i].shape)
            # result_batch[i] = tf.Session().run(
            #     tf.image.random_brightness(X_imgs[i], max_delta= self.config.auggmentation_brightness_max_delta))
            result_batch[i] = sess.run(self.model.brightness_node,
                                       {self.model.X: [X_imgs[i]]})
        # print('result image number (',i,')s shape is: ',result_batch[i].shape)

        # print('Brightening done with shape,' ,len(result_batch))

        return result_batch

    def apply_contrast(self, X_imgs, sess=None):
        # print('input_batch_shape: ',X_imgs.shape)
        result_batch = deepcopy(X_imgs)
        # print('result_batch_shape: ',result_batch.shape)
        for i in range(X_imgs.shape[0]):
            # print('image number (',i,')s shape is: ',X_imgs[i].shape)
            # result_batch[i] = tf.Session().run(tf.image.random_contrast(X_imgs[i], lower= self.config.auggmentation_contrast_lower,
            #                                                             upper= self.config.auggmentation_contrast_upper))
            result_batch[i] = sess.run(self.model.contrast_node,
                                       {self.model.X: [X_imgs[i]]})
            # print('result image number (',i,')s shape is: ',result_batch[i].shape)

        # print('Brightening done with shape,' ,len(result_batch))

        return result_batch

    def add_salt_pepper_noise(self, X_imgs):

        # Need to pr oduce a copy as to not modify the original image

        X_imgs_copy = deepcopy(X_imgs)
        # row, col, _ = X_imgs_copy[0].shape
        salt_vs_pepper = self.config.auggmentation_noise_salt_and_pepper_ratio
        amount = 0.004
        num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
        num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

        for X_img in X_imgs_copy:
            # Add Salt noise
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
            X_img[coords[0], coords[1], :] = 0.0

            # Add Pepper noise
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
            X_img[coords[0], coords[1], :] = 255.0

        return X_imgs_copy

    def add_blur(self, batch):
        blur_factor = random.randint(2, self.config.auggmentation_blur_max_factor)
        kernel = np.ones((blur_factor, blur_factor), np.float32) / (blur_factor * blur_factor)
        result_batch = []
        for i in range(batch.shape[0]):
            dst = cv2.filter2D(batch[i], -1, kernel)
            result_batch.append(dst)
        result_batch = np.array(result_batch)
        # batch = np.concatenate((batch, result_batch), axis=0)
        return result_batch
