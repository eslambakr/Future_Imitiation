import numpy as np
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import random
from copy import deepcopy
class augmentation:
    def __init__(self, config=None):
        """
            apply_auggmentation = False
            auggmentation_counts = 4
            apply_auggmentation_every = 100
            auggmentation_blur = True
            auggmentation_blur_max_factor = 10
            auggmentation_contrast = True
            auggmentation_contrast_lower = 0.2
            auggmentation_contrast_upper = 1.8
            auggmentation_brightness = True
            auggmentation_brightness_max_delta = 100.0
            auggmentation_noise_salt_and_pepper = True
            auggmentation_noise_salt_and_pepper_ratio = 0.5
        :param config:
        """
        self.config=config
    def augment(self, batch,sess=None):
        res = deepcopy(batch[:, :, :, :])
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

    def apply_brightness(self, X_imgs, sess=None):
        result_batch = deepcopy(X_imgs)
        for i in range(X_imgs.shape[0]):
            result_batch[i] = sess.run(self.model.brightness_node,
                                       {self.model.X: [X_imgs[i]]})
        return result_batch

    def apply_contrast(self, X_imgs, sess=None):
        result_batch = deepcopy(X_imgs)
        for i in range(X_imgs.shape[0]):
            result_batch[i] = sess.run(self.model.contrast_node,
                                       {self.model.X: [X_imgs[i]]})


        return result_batch

    def add_salt_pepper_noise(self, X_imgs):

        # Need to produce a copy as to not modify the original image
        X_imgs_copy = deepcopy(X_imgs)
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
        return result_batch
