from PIL import Image, ImageChops
import numpy as np
import string
import os


TAGS_FILE_NAME = 'sample.out'
NUM_TAGS = 200
TOTAL_TESTS = 800
CATEGORIES = [c for c in string.ascii_lowercase if c != 'f' and c != 'h' and c != 'x']

HEIGHT = 70
WIDTH = 100
TARGET_SIZE = 18
SPLITS = 6
COLOR_THRESHOLD = 200
SEED = 42

INPUT_DIM = TARGET_SIZE * TARGET_SIZE
OUTPUT_DIM = len(CATEGORIES)


def _cut_sample(img, bg, split):
    bbox = (split * WIDTH, 0, (split + 1) * WIDTH, HEIGHT)
    cut = img.crop(bbox)
    diff = ImageChops.difference(bg, cut)
    cut = cut.crop(diff.getbbox()).resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)

    data = np.array(cut)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] > COLOR_THRESHOLD:
                data[i][j] = 255
            else:
                data[i][j] = 0
    return Image.fromarray(data)


def split_image(img):
    bg = Image.new(img.mode, (WIDTH, HEIGHT), 255)
    return [_cut_sample(img, bg, split) for split in range(SPLITS)]


def get_image_path(dir, i):
    return os.path.join(dir, '{:03d}.png'.format(i + 1))


def image_to_data(img):
    return np.array(img).reshape(INPUT_DIM) / 255
