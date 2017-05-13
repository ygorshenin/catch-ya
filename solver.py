#!/usr/bin/env python3

from PIL import Image
from keras.models import load_model
import argparse

from utils import *


INPUT_DIM = TARGET_SIZE * TARGET_SIZE


def go(test_dir, model_path):
    model = load_model(model_path)

    tags_file = os.path.join(test_dir, TAGS_FILE_NAME)
    with open(tags_file) as f:
        tags = [line.strip() for line in f.readlines()[0:NUM_TAGS]]

    for i in range(NUM_TAGS, TOTAL_TESTS):
        img = Image.open(get_image_path(test_dir, i))
        X = np.array([image_to_data(sample) for sample in split_image(img)])
        y = model.predict_on_batch(X)
        s = [CATEGORIES[np.argmax(y[j])] for j in range(len(y))]
        tags.append(''.join(s))

    for t in tags:
        print(t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate answers via trained model')
    parser.add_argument('--test_dir',
                        dest='test_dir',
                        help='path to a directory with raw data')
    parser.add_argument('--model_path',
                        dest='model_path',
                        help='path to a file to save model')
    args = parser.parse_args()
    go(args.test_dir, args.model_path)
