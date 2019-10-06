

class Config:
    absolute_path = '/media/eslam/426b7820-cb81-4c46-9430-be5429970ddb/home/eslam/Future_Imitiation/Imitation_Learning/weights'
    summary_dir = '/summaries/'
    ##########################################################
    #               TRAINING OR TESTING CONFIG               #
    ##########################################################
    intel_data = True
    exp_name = absolute_path + '/experiments/stacking_4_previous'
    give_car_push = True
    speed_input = True
    separate_throttle_brake = True
    if separate_throttle_brake:
        num_of_Actions = 3
    else:
        num_of_Actions = 2
    normalized_input = True
    batch_norm = False
    dropout = False
    is_training = True
    load = False

    num_of_stucked_frames = 50
    # Saving
    max_to_keep = 500
    checkpoint_dir = exp_name + '/checkpoints/'
    loss_filename = exp_name + '/loss_tracking.txt'
    # if no stacking is needed put the value to 0 (eg: stacking_frames = 0)
    stacking_frames = 4
    debugging = False
    # if no clipping is needed put the value to 0 (eg: clip_until = 0)
    clip_until = 0
    img_h = 64 - clip_until
    img_w = 64

    # summary_dir = '/summaries/'
    visualization_dir = '/visualization/'
    saving_visualization_img_right_path = visualization_dir + 'vis_right'
    saving_visualization_img_straight_path = visualization_dir + 'vis_straight'
    saving_visualization_img_left_path = visualization_dir + 'vis_left'

    # Data set Training dir
    if intel_data:
        main_dir = '/media/eslam/426b7820-cb81-4c46-9430-be5429970ddb/home/eslam/Future_Imitiation/Intel_dataset/npys/'
        straight_images_dir = main_dir + 'rgb/Training'
        labels_dir = main_dir + 'labels/Training'
    else:
        main_dir = '/media/eslam/D0FCBC10FCBBEF3A/Path_planning_Paper_Dataset/training_data/'
        straight_images_dir = main_dir + 'images/_out_forward'
        labels_dir = main_dir + 'measurements'

    # Data set Validation dir
    if intel_data:
        validation_dir = '/media/eslam/426b7820-cb81-4c46-9430-be5429970ddb/home/eslam/Future_Imitiation/Intel_dataset/npys/'
        straight_val_dir = validation_dir + 'rgb/Val'
        labels_val_dir = validation_dir + 'labels/Val'
    else:
        validation_dir = '/media/eslam/D0FCBC10FCBBEF3A/Path_planning_Paper_Dataset/validation_data/'
        straight_val_dir = validation_dir + 'images/_out_forward'
        labels_val_dir = validation_dir + 'measurements'

    take_samples = False
    take_sample_from_straight_frames = 50000
    take_sample_from_left_frames = 29000
    take_sample_from_right_frames = 29000

    images_dir = '/test_images/'

    # Training
    epochs = 500
    if is_training:
        batch_size = 128
    else:
        batch_size = 1
    learning_rate = 0.0001
    decay_lr = True

    visualize = False
    apply_auggmentation = False
    auggmentation_counts = 4
    apply_auggmentation_every = 20
    auggmentation_blur = True
    auggmentation_blur_max_factor = 10
    auggmentation_contrast = True
    auggmentation_contrast_lower = 0.2
    auggmentation_contrast_upper = 1.8
    auggmentation_brightness = True
    auggmentation_brightness_max_delta = 100.0
    auggmentation_noise_salt_and_pepper = True
    auggmentation_noise_salt_and_pepper_ratio = 0.5
