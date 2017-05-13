#!/usr/bin/env python3

from PIL import Image
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import argparse
import os

from utils import *


def baseline_model():
    model = Sequential()
    model.add(Dense(INPUT_DIM, input_dim=INPUT_DIM, kernel_initializer='normal'))
    model.add(Dense(OUTPUT_DIM, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def load_train_data(data_dir):
    X = []
    y = []
    for tag, c in enumerate(CATEGORIES):
        dir = os.path.join(data_dir, c)
        for file in os.listdir(dir):
            if not file.endswith('.png'):
                continue

            img = Image.open(os.path.join(dir, file))
            X.append(image_to_data(img))
            y.append(tag)

    return np.array(X), to_categorical(y, num_classes=len(CATEGORIES))


def go(data_dir, model_path):
    np.random.seed(SEED)

    X, y = load_train_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, train_size=0.75)

    model = baseline_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=2)
    scores = model.evaluate(X_test, y_test)
    error = 100 * (1.0 - scores[1])
    print()
    print('Baseline model error: {:.2f}%'.format(error))
    model.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--train_dir',
                        dest='train_dir',
                        help='path to a directory with training data')
    parser.add_argument('--model_path',
                        dest='model_path',
                        help='path to a file to save model')
    args = parser.parse_args()
    go(args.train_dir, args.model_path)
