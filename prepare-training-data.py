#!/usr/bin/env python3

from PIL import Image, ImageChops
import argparse
import numpy as np
import os
import string

from utils import *


def go(input_dir, output_dir):
    tags_file = os.path.join(input_dir, TAGS_FILE_NAME)

    with open(tags_file) as f:
        tags = [line.strip() for line in f.readlines()[0:NUM_TAGS]]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dirs = {}
    counts = {}
    for c in string.ascii_lowercase:
        dir = os.path.join(output_dir, c)
        if not os.path.exists(dir):
            os.mkdir(dir)
        dirs[c] = dir
        counts[c] = 0

    for i in range(NUM_TAGS):
        img = Image.open(get_image_path(input_dir, i))
        samples = split_image(img)
        for j, sample in enumerate(samples):
            tag = tags[i][j]
            sample.save(os.path.join(dirs[tag], '{:03d}.png'.format(counts[tag])))
            counts[tag] += 1
    for k, v in sorted(counts.items()):
        print('{}: {} images'.format(k, v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split training images to corresponding directories')
    parser.add_argument('--input_dir',
                        dest='input_dir',
                        help='path to a directory with raw data')
    parser.add_argument('--output_dir',
                        dest='output_dir',
                        help='path to a directory with training data')
    args = parser.parse_args()
    go(args.input_dir, args.output_dir)
