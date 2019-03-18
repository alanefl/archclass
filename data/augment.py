"""
Helper script for augmenting the architecture styles dataset. It will carry out the following transformations
on all the images of a directory containing images (mapping a single image to 10 images, including the original one):

    1. Flip images across the vertical axis (left-to-right)
    2. Take 4 random crops of the image.
    3. Rotate image 5 and 15 degrees clockwise and counter-clockwise.

"""

import argparse
import random
import os
from tqdm import tqdm
from PIL import Image

# The percentage of the an original image's height and width to use for random crops.
PERCENTAGE = 0.50

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True,
                    help="Directory containing images to augment.")
parser.add_argument('--output_dir', required=True,
                    help="Directory where to place new images.")
args = parser.parse_args()


def get_rand_crop(filepath):
    image = Image.open(filepath)
    width, height = image.size
    crop_width = int(width * PERCENTAGE)
    crop_height = int(height * PERCENTAGE)

    # Choose a random point for the top left corner of the new crop.
    tl_x = random.randint(0, crop_width)
    tl_y = random.randint(0, crop_height)

    coords = (tl_x, tl_y, tl_x + crop_width, tl_y + crop_height)
    return image.crop(coords)


def get_rotated(filepath, angle):
    image = Image.open(filepath)
    return image.rotate(angle)


def get_flipped(filepath):
    image = Image.open(filepath)
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def generate_and_save_augmentations(output_dir, filepath):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Original image
    orig = Image.open(filepath)

    # Flip
    flipped_image = get_flipped(filepath)

    # Rotate
    rotations = []
    for deg in [-15, -5, 5, 15]:
        rotations.append(get_rotated(filepath, deg))

    # Crops
    crops = []
    for _ in range(4):
        crops.append(get_rand_crop(filepath))

    # Save new images.
    filename_no_ext = filepath.split('/')[-1].split('.')[0]

    # -0 => original
    # -1 => flipped image
    # -2-5 => rotations
    # -6-9 => crops
    for idx, img in enumerate([orig, flipped_image] + rotations + crops):
        out_filename = "%s-%d.jpg" % (filename_no_ext, idx)
        out_full_path = os.path.join(output_dir, out_filename)
        img.save(out_full_path)


for filename in tqdm(os.listdir(args.data_dir)):
    if not filename.endswith(".jpg"):
        continue
    generate_and_save_augmentations(args.output_dir, os.path.join(args.data_dir, filename))
