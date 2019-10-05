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


def generate_future_frames(frame_0, frame_1, frame_2, frame_3, num_of_generated_frames, batch_size, debug=False):
    seed = 7
    img_size = 64
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    results_gif_dir = "../results_test/carla_50k"
    results_png_dir = "../results_test/carla_50k"
    dataset_hparams_dict = {}
    model_hparams_dict = {}
    checkpoint = "../logs/carla_intel/ours_savp"
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
    num_stochastic_samples = 1
    fps = 4
    dataset_hparams = "sequence_length = "+str(4+num_of_generated_frames)
    #TODO: should be changed to feed the 4 images directly to it.
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
    output_gif_dir = os.path.join(results_gif_dir, os.path.split(checkpoint_dir)[1])
    output_png_dir = os.path.join(results_png_dir, os.path.split(checkpoint_dir)[1])

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
    model = VideoPredictionModel(
        mode=mode,
        hparams_dict=hparams_dict,
        hparams=None)

    sequence_length = model.hparams.sequence_length
    gif_length = sequence_length
    context_frames = model.hparams.context_frames
    future_length = sequence_length - context_frames
    num_examples_per_epoch = dataset.num_examples_per_epoch()
    if num_examples_per_epoch % batch_size != 0:
        raise ValueError('batch_size should evenly divide the dataset size %d' % num_examples_per_epoch)

    inputs = dataset.make_batch(batch_size)
    input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}
    with tf.variable_scope(''):
        model.build_graph(input_phs)

    for output_dir in (output_gif_dir, output_png_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()
    model.restore(sess, checkpoint)
    sample_ind = 0

    print("evaluation samples from %d to %d" % (sample_ind, sample_ind + batch_size))

    # feed_dict = {input_ph: input_results[name] for name, input_ph in input_phs.items()}
    input_results = np.zeros((batch_size, sequence_length, img_size, img_size, 3))
    input_results[0][0] = frame_0/255
    input_results[0][1] = frame_1/255
    input_results[0][2] = frame_2/255
    input_results[0][3] = frame_3/255
    for name, input_ph in input_phs.items():
        feed_dict = {input_ph: input_results}

    for stochastic_sample_ind in range(num_stochastic_samples):
        gen_images = sess.run(model.outputs['gen_images'], feed_dict=feed_dict)
        # only keep the future frames
        gen_images = gen_images[:, -future_length:]
        if debug:
            for i, gen_images_ in enumerate(gen_images):
                context_images_ = (input_results[i] * 255.0).astype(np.uint8)
                gen_images_ = (gen_images_ * 255.0).astype(np.uint8)

                gen_images_fname = 'gen_image_%05d_%02d.gif' % (sample_ind + i, stochastic_sample_ind)
                context_and_gen_images = list(context_images_[:context_frames]) + list(gen_images_)
                if gif_length:
                    context_and_gen_images = context_and_gen_images[:gif_length]
                save_gif(os.path.join(output_gif_dir, gen_images_fname),
                         context_and_gen_images, fps=fps)

                gen_image_fname_pattern = 'gen_image_%%05d_%%02d_%%0%dd.png' % max(2, len(str(len(gen_images_) - 1)))
                for t, gen_image in enumerate(gen_images_):
                    gen_image_fname = gen_image_fname_pattern % (sample_ind + i, stochastic_sample_ind, t)
                    if gen_image.shape[-1] == 1:
                      gen_image = np.tile(gen_image, (1, 1, 3))
                    else:
                      gen_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_png_dir, gen_image_fname), gen_image)

        sample_ind += batch_size
    return gen_images


def main():
    frame_0 = cv2.cvtColor(cv2.imread("../../frame_0.png"), cv2.COLOR_BGR2RGB)
    frame_1 = cv2.cvtColor(cv2.imread("../../frame_1.png"), cv2.COLOR_BGR2RGB)
    frame_2 = cv2.cvtColor(cv2.imread("../../frame_2.png"), cv2.COLOR_BGR2RGB)
    frame_3 = cv2.cvtColor(cv2.imread("../../frame_3.png"), cv2.COLOR_BGR2RGB)
    gen_images = generate_future_frames(frame_0, frame_1, frame_2, frame_3, 4, 1)
    print(gen_images.shape)


if __name__ == '__main__':
    main()
