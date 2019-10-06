import numpy as np
from utils.lidar_postprocessing import postprocess_lidar
from PIL import Image
from tqdm import tqdm
from utils.carla_to_cityscapes_segmentation_mapping import map_original, seg_to_cityscape_rgb
from copy import deepcopy
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from utils.img_utils import decode_labels

from utils.lidar_pgm import PGM

counter = 0
directions = ["straight", "right", "left"]
for x in directions:
    direction = x
    path = '/media/user/research_data1/CARLA/LiDAR/lidar_data_combined_with_PGM/' + str(direction) + '/'
    # path = '/media/user/research_data1/CARLA/LiDAR/rich_pix2pix_data_no_pedestrians/straight4'
    saving_path = '/media/user/research_data1/CARLA/LiDAR/lidar_data_combined_with_PGM/alldata_sep/'
    # saving_path = '/media/user/research_data1/CARLA/LiDAR/rich_pix2pix_data_no_pedestrians/'
    saving_path_rgb = saving_path + 'rgb/'+x+'/'
    saving_path_seg = saving_path + 'seg/'+x+'/'
    saving_path_lidar = saving_path + 'lidar/'+x+'/'
    saving_path_pgm = saving_path+'PGM/'+x+'/'
    # saving_path_depth = path+'depth_samples/'
    rgb = np.load(path + 'rgb.npy')
    # depth = np.load(path + 'depth.npy')
    seg = np.load(path + 'seg.npy')
    lidar = np.load(path + 'LiDAR.npy')
    pgm = np.load(path + 'PGM.npy')
    for i in tqdm(range(seg.shape[0])):
        idx = i
        label_colours_carla_13 = [(192, 0, 0),
                                  (0, 128, 0),
                                  (192, 128, 128),
                                  (0, 0, 128),
                                  (64, 0, 128),
                                  (128, 64, 0),
                                  (128, 192, 0),
                                  (0, 64, 128),
                                  (0, 0, 0),
                                  (255, 255, 0),
                                  (0, 255, 255),
                                  (0, 255, 0),
                                  (255, 0, 255)]

        label_colours_scala_3 = [(0, 0, 0),
                                 (250, 250, 0),
                                 (69, 179, 231)]


        def seg_color(mask, num_classes):
            """Decode batch of segmentation masks.

            Args:
              mask: result of inference after taking argmax.
              num_images: number of images to decode from the batch.
              num_classes: number of classes to predict (including background).

            Returns:
              A batch with num_images RGB images of the same size as the input.
            """
            colours = None
            if num_classes == 13:
                colours = label_colours_carla_13
            elif num_classes == 3:
                colours = label_colours_scala_3
            else:
                print("ERROR this number of classes don't have a defined colours")
                exit(-1)

            n, h, w = mask.shape
            outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
            for i in range(n):
                img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
                pixels = img.load()
                for j_, j in enumerate(mask[i, :, :]):
                    for k_, k in enumerate(j):
                        if k < num_classes:
                            pixels[k_, j_] = colours[k]
                outputs[i] = np.array(img)
            return outputs


        from scipy import misc

        ### RGB
        misc.imsave(saving_path_rgb + str(idx) + '.png', rgb[i])
        # ### DEPTH
        # depth_img = depth[idx]
        # upper = depth_img > 0.1
        # lower = depth_img <= 0.1
        # depth_img[upper] = 1.0
        # depth_img[lower] *= 10.0
        # # print(depth_img.shape)
        # misc.imsave(saving_path_depth + str(idx) + '.png', depth_img)
        # ### seg
        seg_img = seg[i].astype(np.uint8)
        # seg = decode_labels(np.expand_dims(seg,0),3)
        # Original segmentation conversion for Carla
        seg_img = seg_color(np.expand_dims(seg_img, 0), 13)
        misc.imsave(saving_path_seg + str(idx) + '.png', seg_img[0])
        # exit(0)
        # Carla to cityscapes conversion
        # seg_img_copy = deepcopy(seg_to_cityscape_rgb(map_original(seg_img)))
        # misc.imsave(saving_path_seg + str(idx) + '.png', seg_img_copy)
        # seg_img = None

        # print(seg_img.shape)
        # ### Lidar
        # LiDAR_to_RGB_hlimit = 1750  # parser.add_argument('--hlimit', type=int, default=1750)
        # LiDAR_to_RGB_vlimit = 4000  # parser.add_argument('--vlimit', type=int, default=4000)
        # LiDAR_to_RGB_clip_back = True  # parser.add_argument('--clip_back', type=bool, default=True)
        # LiDAR_to_RGB_resolution = 0.05  # parser.add_argument('--resolution', type=float, default=0.15)
        # LiDAR_to_RGB_thickness = 0  # par
        lidar_img = lidar[i]
        lidar_img = cv2.resize(lidar_img, (200, 88), interpolation=cv2.INTER_NEAREST)
        misc.imsave(saving_path_lidar + str(idx) + '.png', lidar_img * 255.0)
        #
        ### PGM
        pgm_out = pgm[idx]
        # _
        misc.imsave(saving_path_pgm + str(idx) + '.png',pgm_out)
        # print('Done')
        counter += 1
