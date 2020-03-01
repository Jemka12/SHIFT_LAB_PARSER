from pathlib import Path
import pandas as pd
import tensorflow as tf
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile
from pathlib import Path


def predict(model, path_img):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=True,
        featurewise_std_normalization=True)
    img_height, img_width = 224, 224
    batch_size = 40
    datagen.mean = np.array([0.485, 0.456, 0.406])
    datagen.std = np.array([0.229, 0.224, 0.225])

    datagen_generator = datagen.flow_from_directory(
        path_img,
        shuffle=False,
        color_mode="rgb",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')  # set as training data

    Y_pred = model.predict(datagen_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    TEST_DIR = Path(path_img)
    test_files = sorted(list(TEST_DIR.rglob('*.*')))

    data_file = pd.DataFrame(test_files)
    data_file['class'] = y_pred
    return data_file


def distribution(classes, data_file):
    for i in classes:
        try:
            os.mkdir(i)
        except FileExistsError:
            pass
    for pathh, classs in data_file.iterrows():
        copyfile(classs[0], Path(str(classs['class']) + '/' + str(pathh) + '.jpg'))