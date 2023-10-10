import numpy as np
from PIL import Image, ImageChops


def transform(image):
    #import matplotlib.pyplot as plt
    # black image
    sh = image.size
    img = np.zeros((sh[0], sh[1], 3), dtype=np.uint8)
    # show
    #plt.imshow(img)

    # make it green
    def update_pixel(p):
        return (cityscapes_palette[p[0]])

    transformed_img = np.apply_along_axis(update_pixel, 2, image)
    # show
    #plt.imshow(transformed_img)
    return Image.fromarray(transformed_img)


def absolute_difference(image1, image2):
    """
    Compute the pixelwise absolute difference between two RGBA PIL images.

    Args:
    image1 (PIL.Image.Image): The first input image.
    image2 (PIL.Image.Image): The second input image.

    Returns:
    PIL.Image.Image: A new image representing the absolute difference.
    """

    # Ensure that both images have the same size and mode (RGBA)
    if image1.size != image2.size or image1.mode != 'RGBA' or image2.mode != 'RGBA':
        raise ValueError("Input images must have equal size and RGBA mode")

    # Convert only the RGB channels of the images to NumPy arrays
    img1_array = np.array(image1, dtype=int)[:, :, :3]
    img2_array = np.array(image2, dtype=int)[:, :, :3]

    # Compute the pixel-wise absolute difference (excluding the alpha channel)
    sh = image1.size
    #diff_array = np.zeros((sh[0], sh[1], 3), dtype=np.int16)
    diff_array = np.abs(img1_array - img2_array)

    # Convert the NumPy array back to a PIL image
    diff_image = Image.fromarray(diff_array, 'RGB')

    return diff_image



cityscapes_palette = {
    0: (0, 0, 0), # unlabeled = 0                                                            # cityscape
    1: (128, 64, 128), # road = 1
    2: (244, 35, 232), # sidewalk = 2
    3: (70, 70, 70), # building = 3
    4: (102, 102, 156), # wall = 4
    5: (190, 153, 153), # fence = 5
    6: (153, 153, 153), # pole = 6
    7: (250, 170, 30), # traffic light = 7
    8: (220, 220, 0), # traffic sign = 8
    9: (107, 142, 35), # vegetation = 9
    10: (152, 251, 152), # terrain = 10
    11: (70, 130, 180), # sky = 11
    12: (220, 20, 60), # pedestrian = 12
    13: (255, 0, 0), # rider = 13
    14: (0, 0, 142), # Car = 14
    15: (0, 0, 70), # trck = 15
    16: (0, 60, 100), # bs = 16
    17: (0, 80, 100), # train = 17
    18: (0, 0, 230), # motorcycle = 18
    19: (119, 11, 32), # bicycle = 19
    # custom
    20: (110, 190, 160), # static = 20
    21: (170, 120, 50), # dynamic = 21
    22: (55, 90, 80), # other = 22
    23: (45, 60, 150), # water = 23
    24: (157, 234, 50), # road line = 24
    25: (81, 0, 81), # grond = 25
    26: (150, 100, 100), # bridge = 26
    27: (230, 150, 140), # rail track = 27
    28: (180, 165, 180), # gard rail = 28
    29: (255, 000, 000), # mltirotor = 29
    30: (000, 255, 000), # fixedwing = 30
    31: (000, 255, 255), # airliner = 31
    32: (255, 000, 255), # bird = 32
}


def convert_image_with_palette(image, color_palette=cityscapes_palette):
    # Convert the PIL image to a NumPy array
    image_array = np.array(image)

    # Create an empty NumPy array to hold the converted image
    converted_image = np.zeros_like(image_array, dtype=np.uint8)

    # Iterate through the color palette and map pixel values
    for label, color in color_palette.items():
        class_image = image_array[:,:,0]
        mask = (class_image == label)
        if label >= 29:
            print(f"label {label} found in {np.count_nonzero(mask)} pixels")
            converted_image[mask] = (color[0], color[1], color[2], 255)
        else:
            converted_image[mask] = (color[0]/4, color[1]/4, color[2]/4, 255)

    # Convert the NumPy array back to a PIL image
    converted_image = Image.fromarray(converted_image)

    return converted_image