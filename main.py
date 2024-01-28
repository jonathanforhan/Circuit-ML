#!/usr/bin/python3
from PIL import Image
import tkinter as tk
from keras import layers
import keras
import numpy as np
import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMG_SIZE = (120, 120)
EPOCHS = 100
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
    x, y = [], []
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


if 'data.model'not in os.listdir():
    train_model()

model = keras.models.load_model('data.model')
if model is None:
    print('failed to load model')
    exit(-1)


class Painter:
    def __init__(self, win: tk.Tk):
        self.win = win
        self.canvas = tk.Canvas(win, width=600, height=600, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.old_x = None
        self.old_y = None

        menu = tk.Menu(win)
        win.config(menu=menu)
        opts = tk.Menu(menu)
        menu.add_cascade(label='Menu', menu=opts)
        opts.add_command(label='Clear', command=self.clear)
        opts.add_command(label='Evaluate', command=self.evaluate)

        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def paint(self, e):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, e.x, e.y,
                                    width=18, fill='black', capstyle='round', smooth=True)
        self.old_x, self.old_y = e.x, e.y

    def reset(self, _):
        self.old_x, self.old_y = None, None

    def clear(self):
        self.canvas.delete(tk.ALL)

    def evaluate(self):
        self.save_img()
        img_arr = cv2.imread('contents.png')
        img_arr = cv2.resize(img_arr, IMG_SIZE)
        x = np.invert(img_arr)
        predictions = model.predict(np.array([x]))
        index = np.argmax(predictions)
        popup = tk.Toplevel(self.win)
        popup.geometry('500x200')
        popup.title('Model Prediction')
        tk.Label(popup, text=f'Prediction is {class_names[index]}').place(
            x=100, y=70)

    def save_img(self):
        fname = 'contents'
        self.canvas.postscript(file=fname+'.eps')
        img = Image.open(fname+'.eps')
        img.save(fname+'.png', 'png')


win = tk.Tk()
win.resizable(False, False)
win.title("Circuit-ML")
_ = Painter(win)
win.mainloop()
