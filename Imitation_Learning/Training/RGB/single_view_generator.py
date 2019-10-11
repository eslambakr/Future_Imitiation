import numpy as np
from os import listdir
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


class DataLoader:
    def __init__(self, config, future_generator, is_training):
        self.config = config
        self.episode_num = None
        # Images
        if is_training:
            self.forward_files = listdir(self.config.straight_images_dir)
        else:  # Validation
            self.forward_files = listdir(self.config.straight_val_dir)

        self.forward_files.sort()
        self.forward_images = []
        self.forward_temp = None
        self.measurements_per_episode_temp = None
        # measurements
        if is_training:
            self.measurements_files = listdir(self.config.labels_dir)
        else:
            self.measurements_files = listdir(self.config.labels_val_dir)
        self.measurements_files.sort()
        self.measurements_per_episode = []
        measurements = []
        self.num_of_episodes = len(self.measurements_files)
        self.total_size = 0
        for file in self.measurements_files:
            measurements.append(np.load(self.config.labels_dir + '/' + file))
            self.total_size += len(measurements[-1])
        print("Loading labels done ")
        self.future_generator = future_generator


    def process_data(self, data):
        if self.config.normalized_input and False:
            print('Normalizing data')
            data = data.astype(np.float32) / np.max(data).astype(np.float32)
        return data

    def get_batch(self, episode_num, item_nums):
        self.forward_images = []
        self.measurements_per_episode = []
        if self.episode_num != episode_num:
            self.episode_num = episode_num
            self.forward_temp = np.load(self.config.straight_images_dir + '/' + self.forward_files[episode_num])
            self.measurements_per_episode_temp = np.load(
                self.config.labels_dir + '/' + self.measurements_files[episode_num])
        for item_num in item_nums:
            if self.config.p_stacking_frames:
                for i in range(self.config.p_stacking_frames):
                    dummy_index = item_num - self.config.p_stacking_frames + 1 + i
                    self.forward_images.append(
                        self.forward_temp[dummy_index if dummy_index >= 0 else 0][self.config.clip_until:, :, :])
            if self.config.f_stacking_frames:
                gen_images = self.future_generator.generate_future_frames(self.forward_images[-4],
                                                                          self.forward_images[-3],
                                                                          self.forward_images[-2],
                                                                          self.forward_images[-1],
                                                                          debug=False)
                for future_conter in range(self.config.f_stacking_frames):
                    self.forward_images.append(cv2.resize(gen_images[0, future_conter],
                                                          (self.config.img_h, self.config.img_h)))
            else:
                self.forward_images.append(self.forward_temp[item_num][self.config.clip_until:, :, :])
            self.measurements_per_episode.append(self.measurements_per_episode_temp[item_num])
        self.measurements_per_episode = np.asarray(self.measurements_per_episode)
        if self.config.separate_throttle_brake and not self.config.speed_input:
            return self.forward_images, self.measurements_per_episode[:, :3], self.measurements_per_episode[:, 4]
        if self.config.separate_throttle_brake and self.config.speed_input:
            return self.forward_images, self.measurements_per_episode, self.measurements_per_episode[:, 4]
        else:
            # TODO: cover this case {if not self.config.separate_throttle_brake and self.config.speed_input}
            return self.forward_images, self.measurements_per_episode[:, :2], self.measurements_per_episode[:, 4]
