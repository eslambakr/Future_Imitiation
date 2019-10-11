from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import errno
import json
import os
import random
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import tensorflow as tf

from video_prediction import datasets, models
from video_prediction.utils.ffmpeg_gif import save_gif

class inference_future_generator:
    def __init__(self, num_of_generated_frames, batch_size):
        seed = 7
        self.img_size = 64
        self.batch_size = batch_size
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        results_gif_dir = "../results_test/carla_50k"
        results_png_dir = "../results_test/carla_50k"
        dataset_hparams_dict = {}
        model_hparams_dict = {}
        checkpoint = "/media/eslam/426b7820-cb81-4c46-9430-be5429970ddb/home/eslam/Future_Imitiation/video_prediction-master/logs/carla_intel/ours_savp"
        # loading weights
        checkpoint_dir = os.path.normpath(checkpoint)
        if not os.path.isdir(checkpoint):
            checkpoint_dir, _ = os.path.split(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)

        dataset = "carla_intel"
        model = "savp"
        mode = "test"
        num_epochs = 1
        gpu_mem_frac = 0
        self.num_stochastic_samples = 1
        self.fps = 4
        dataset_hparams = "sequence_length = " + str(4 + num_of_generated_frames)
        # TODO: should be changed to feed the 4 images directly to it.
        input_dir = "/media/eslam/426b7820-cb81-4c46-9430-be5429970ddb/home/eslam/Future_Imitiation/Intel_dataset/tf_record/test"
        try:
            with open(os.path.join(checkpoint_dir, "dataset_hparams.json")) as f:
                dataset_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("dataset_hparams.json was not loaded because it does not exist")
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")
        self.output_gif_dir = os.path.join(results_gif_dir, os.path.split(checkpoint_dir)[1])
        self.output_png_dir = os.path.join(results_png_dir, os.path.split(checkpoint_dir)[1])

        VideoDataset = datasets.get_dataset_class(dataset)
        dataset = VideoDataset(
            input_dir,
            mode=mode,
            num_epochs=num_epochs,
            seed=seed,
            hparams_dict=dataset_hparams_dict,
            hparams=dataset_hparams)

        VideoPredictionModel = models.get_model_class(model)
        hparams_dict = dict(model_hparams_dict)
        hparams_dict.update({
            'context_frames': dataset.hparams.context_frames,
            'sequence_length': dataset.hparams.sequence_length,
            'repeat': dataset.hparams.time_shift,
        })
        self.model = VideoPredictionModel(
            mode=mode,
            hparams_dict=hparams_dict,
            hparams=None)

        self.sequence_length = self.model.hparams.sequence_length
        self.gif_length = self.sequence_length
        context_frames = self.model.hparams.context_frames
        self.future_length = self.sequence_length - context_frames
        num_examples_per_epoch = dataset.num_examples_per_epoch()
        if num_examples_per_epoch % self.batch_size != 0:
            raise ValueError('batch_size should evenly divide the dataset size %d' % num_examples_per_epoch)

        inputs = dataset.make_batch(self.batch_size)
        self.input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}
        with tf.variable_scope(''):
            self.model.build_graph(self.input_phs)

        for output_dir in (self.output_gif_dir, self.output_png_dir):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.graph.as_default()
        self.model.restore(self.sess, checkpoint)
        self.sample_ind = 0

    def generate_future_frames(self, frame_0, frame_1, frame_2, frame_3, debug=False):
        input_results = np.zeros((self.batch_size, self.sequence_length, self.img_size, self.img_size, 3))
        input_results[0][0] = cv2.resize(frame_0, (self.img_size, self.img_size))/255
        input_results[0][1] = cv2.resize(frame_1, (self.img_size, self.img_size))/255
        input_results[0][2] = cv2.resize(frame_2, (self.img_size, self.img_size))/255
        input_results[0][3] = cv2.resize(frame_3, (self.img_size, self.img_size))/255
        for name, input_ph in self.input_phs.items():
            feed_dict = {input_ph: input_results}

        for stochastic_sample_ind in range(self.num_stochastic_samples):
            gen_images = self.sess.run(self.model.outputs['gen_images'], feed_dict=feed_dict)
            # only keep the future frames
            gen_images = gen_images[:, -self.future_length:]
            if debug:
                for i, gen_images_ in enumerate(gen_images):
                    context_images_ = (input_results[i] * 255.0).astype(np.uint8)
                    gen_images_ = (gen_images_ * 255.0).astype(np.uint8)

                    gen_images_fname = 'gen_image_%05d_%02d.gif' % (self.sample_ind + i, stochastic_sample_ind)
                    context_and_gen_images = list(context_images_[:context_frames]) + list(gen_images_)
                    if self.gif_length:
                        context_and_gen_images = context_and_gen_images[:self.gif_length]
                    save_gif(os.path.join(self.output_gif_dir, gen_images_fname),
                             context_and_gen_images, fps=self.fps)

                    gen_image_fname_pattern = 'gen_image_%%05d_%%02d_%%0%dd.png' % max(2, len(str(len(gen_images_) - 1)))
                    for t, gen_image in enumerate(gen_images_):
                        gen_image_fname = gen_image_fname_pattern % (self.sample_ind + i, stochastic_sample_ind, t)
                        if gen_image.shape[-1] == 1:
                          gen_image = np.tile(gen_image, (1, 1, 3))
                        else:
                          gen_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(self.output_png_dir, gen_image_fname), gen_image)

            self.sample_ind += self.batch_size
        return gen_images


def main():
    frame_0 = cv2.cvtColor(cv2.imread("../../frame_0.png"), cv2.COLOR_BGR2RGB)
    frame_1 = cv2.cvtColor(cv2.imread("../../frame_1.png"), cv2.COLOR_BGR2RGB)
    frame_2 = cv2.cvtColor(cv2.imread("../../frame_2.png"), cv2.COLOR_BGR2RGB)
    frame_3 = cv2.cvtColor(cv2.imread("../../frame_3.png"), cv2.COLOR_BGR2RGB)
    future_generator = inference_future_generator(num_of_generated_frames=4, batch_size=1)
    gen_images = future_generator.generate_future_frames(frame_0, frame_1, frame_2, frame_3, debug=True)
    print(gen_images.shape)


if __name__ == '__main__':
    main()
