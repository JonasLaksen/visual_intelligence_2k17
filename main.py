import random

from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Lambda, regularizers
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import time
from keras.callbacks import CSVLogger
import csv
import cv2
import numpy as np
csv_logger = CSVLogger(time.strftime("./logs/%m-%d-%H:%M:%S"), append=True, separator=';')
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def preprocess_image(x, flip=False):
    x=cv2.imread(x,0)
    if flip:
        x = cv2.flip(x,1)
    x=x[50:150,0:320].copy()
    x=cv2.resize(x, (160,50))
    x=x.reshape((50,160,1))
    return x/255.0

def plot(y):
    y, binEdges = np.histogram(y, bins=100)
    bincenters = .5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters, y)
    plt.title('Frequency of steering angles')
    plt.xlabel('Steering angle')
    plt.ylabel('Frequency')
    plt.show()

def generate_samples():
    X,y = [],[]
    with open('driving_log.csv') as csvfile:
        for center_image,_,_,steering_angle,_,_, speed in csv.reader(csvfile):
            if float(steering_angle)**2 > 0.02 or float(steering_angle)**2 + .02 > random.random():
                X.append(preprocess_image(center_image))
                y.append(float(steering_angle))
                if float(steering_angle)**2 > .02:
                    X.append(preprocess_image(center_image, True))
                    y.append(-float(steering_angle))

    return np.asarray(X),np.asarray(y)

X, y = generate_samples()

model = Sequential([
    Conv2D(24, strides=2, kernel_size=5, kernel_regularizer=regularizers.l2(.001), input_shape=(50, 160, 1)),
    Conv2D(36, strides=2, kernel_size=5, kernel_regularizer=regularizers.l2(.001)),
    Conv2D(48, strides=1, kernel_size=5, kernel_regularizer=regularizers.l2(.001)),
    Conv2D(64, strides=1, kernel_size=2, kernel_regularizer=regularizers.l2(.001)),
    Conv2D(64, strides=1, kernel_size=3, kernel_regularizer=regularizers.l2(.001)),
    Flatten(),
    Dense(100, activation='elu', kernel_regularizer=regularizers.l2(.001)),
    Dense(50, activation='elu', kernel_regularizer=regularizers.l2(.001)),
    Dense(10, activation='elu', kernel_regularizer=regularizers.l2(.001)),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer=Adam())
history = model.fit(X,y,batch_size=32,epochs=50,callbacks=[csv_logger], validation_split=0.1, verbose=2)
model.save('model.h5')
