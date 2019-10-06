from PIL import Image
import numpy as np

# colour map

label_colours_global = [(0, 0, 0),
                        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
label_colours_scala_1 = [(192, 0, 0)]
label_colours_scala_2 = [(0, 0, 0),(255,255,255)]

label_colours_scala_3 = [(0, 0, 0),
                         (255, 255, 0),
                         (52, 152, 219)]

label_colours_scala_5 = [(192, 0, 0),
                         (0, 128, 0),
                         (0, 0, 128),
                         (64, 0, 128),
                         (0, 0, 0)]

label_colours_scala_6 = [(192, 0, 0),
                         (0, 128, 0),
                         (192, 128, 128),
                         (0, 0, 128),
                         (64, 0, 128),
                         (0, 0, 0)]

label_colours_scala_7 = [(192, 0, 0),
                         (0, 128, 0),
                         (192, 128, 128),
                         (0, 0, 128),
                         (64, 0, 128),
                         (128, 64, 0),
                         (0, 0, 0)]

label_colours_scala_9 = [(192, 0, 0),
                         (0, 128, 0),
                         (192, 128, 128),
                         (0, 0, 128),
                         (64, 0, 128),
                         (128, 64, 0),
                         (128, 192, 0),
                         (0, 64, 128),
                         (0, 0, 0)]

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


def decode_labels(mask, num_classes):
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
    elif num_classes == 7:
        colours = label_colours_scala_7
    elif num_classes == 6:
        colours = label_colours_scala_6
    elif num_classes == 5:
        colours = label_colours_scala_5
    elif num_classes == 9:
        colours = label_colours_scala_9
    elif num_classes == 3:
        colours = label_colours_scala_3
    elif num_classes == 1:
        colours = label_colours_scala_1
    elif num_classes == 2:
        colours = label_colours_scala_2

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
