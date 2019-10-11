class Config:
    """
    ###########################################################
    #                  DATA COLLECTION CONFIG                 #
    ###########################################################
    training_mes_dir = 'training_data/measurements'
    training_forward_img_dir = 'training_data/images/_out_forward'
    training_right_img_dir = 'training_data/images/_out_right'
    training_left_img_dir = 'training_data/images/_out_left'
    training_back_img_dir = 'training_data/images/_out_back'

    validation_mes_dir = 'validation_data/measurements'
    validation_forward_img_dir = 'validation_data/images/_out_forward'
    validation_right_img_dir = 'validation_data/images/_out_right'
    validation_left_img_dir = 'validation_data/images/_out_left'
    validation_back_img_dir = 'validation_data/images/_out_back'

    testing_mes_dir = 'testing_data/measurements'
    testing_forward_img_dir = 'testing_data/images/_out_forward'
    testing_right_img_dir = 'testing_data/images/_out_right'
    testing_left_img_dir = 'testing_data/images/_out_left'
    testing_back_img_dir = 'testing_data/images/_out_back'

    # types of data: 1 = training data / 2 = validation data / 3 = testing data
    type_of_data = 2
    if type_of_data == 1:
        saving_mes_dir = training_mes_dir
        saving_forward_img_dir = training_forward_img_dir
        saving_right_img_dir = training_right_img_dir
        saving_left_img_dir = training_left_img_dir
        saving_back_img_dir = training_back_img_dir
    elif type_of_data == 2:
        saving_mes_dir = validation_mes_dir
        saving_forward_img_dir = validation_forward_img_dir
        saving_right_img_dir = validation_right_img_dir
        saving_left_img_dir = validation_left_img_dir
        saving_back_img_dir = validation_back_img_dir
    else:
        saving_mes_dir = testing_mes_dir
        saving_forward_img_dir = testing_forward_img_dir
        saving_right_img_dir = testing_right_img_dir
        saving_left_img_dir = testing_left_img_dir
        saving_back_img_dir = testing_back_img_dir
    num_of_collected_frames = 60000
    num_of_spawn_cars = 120
    target_speed = 11
    num_of_stucked_frames = 50
    start_episode_num = 64
    start_frame_num = 10201
    moving_cars = False
    """
    ##########################################################
    #                     Bench Marking                      #
    ##########################################################
    speed_input = True
    p_stacking_frames = 4
    f_stacking_frames = 4
    separate_throttle_brake = True
    if separate_throttle_brake:
        num_of_Actions = 3
    else:
        num_of_Actions = 2

    # if no clipping is needed put the value to 0 (eg: clip_until = 0)
    clip_until = 0
    img_h = 200 - clip_until
    img_w = 200

    normalized_input = True
    batch_norm = False
    dropout = False
    is_training = False
    load = True
    checkpoint_dir = "/media/eslam/426b7820-cb81-4c46-9430-be5429970ddb/home/eslam/Future_Imitiation/Imitation_Learning/weights/experiments/stacking_4_previous_200/checkpoints"
    give_car_push = True
    # Training
    epochs = 500
    if is_training:
        batch_size = 128
    else:
        batch_size = 1
    learning_rate = 0.0001
    decay_lr = True
    # Saving
    max_to_keep = 500
    """
    absolute_path = '/home/eslam/Masters/Advanced_NN/CARLA_0.9.4/Path_Planning_using_Deeplearning/Training/RGB'
    summary_dir = '/summaries/'
    
    logger_path = "/home/eslam/Masters/Advanced_NN/CARLA_0.9.4/Path_Planning_using_Deeplearning/Testing/logger_forward.txt"
    exp_name = absolute_path + '/experiments/rgb_stacked_4_speed'
    checkpoint_dir = exp_name + '/checkpoints/'
    loss_filename = exp_name + '/loss_tracking.txt'
    """

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
