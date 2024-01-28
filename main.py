#!/usr/bin/python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers

IMG_SIZE = (120, 120)
EPOCHS = 24
TRAIN_DIR = 'train'
TEST_DIR = 'test'

class_names = [
    'ac_src',
    'Ammeter',
    'battery',
    'cap',
    'curr_src',
    'dc_volt_src_1',
    'dc_volt_src_2',
    'dep_curr_src',
    'dep_volt',
    'diode',
    'gnd_1',
    'gnd_2',
    'inductor',
    'resistor',
    'voltmeter',
]

def import_dataset(path: str, invert=False):
    x, y= [], []
    for dir in os.listdir(path):
        sub_path = path + '/' + dir
        sub_dir = os.listdir(sub_path)
        for img in sub_dir:
            image_path = sub_path + '/' + img
            img_arr = cv2.imread(image_path)
            img_arr = cv2.resize(img_arr, IMG_SIZE)
            img_arr = np.invert(img_arr) if invert else img_arr
            x.append(img_arr)
            y.append(class_names.index(dir))

    x = np.array(x) / 255.0
    y = np.array(y)
    return (x, y)


def train_model():
    x_train, y_train = import_dataset(TRAIN_DIR)
    x_test, y_test = import_dataset(TRAIN_DIR)

    model = keras.Sequential([
        layers.Input((120, 120, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(class_names), activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        shuffle=True
    )

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"loss: {loss}, accuracy: {accuracy}")

    model.save('data.model')


if not 'data.model' in os.listdir():
    train_model()

model = keras.models.load_model('data.model')
if model == None:
    print('failed to load model')
    exit(-1)

x, y = import_dataset(TEST_DIR, invert=True)

for i in range(len(x)):
    predictions = model.predict(np.array([x[i]]))
    index = np.argmax(predictions)
    print(f'Prediction is {class_names[index]}')
    print(f'Reality is {class_names[y[i]]}')
