import random

from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Lambda, regularizers
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import time
from keras.callbacks import CSVLogger
import csv
import cv2
import numpy as np

from preprocessing import preprocess_image

csv_logger = CSVLogger(time.strftime("./logs/%m-%d-%H:%M:%S"), append=True, separator=';')
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
epochs=50


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
    counter = 0
    with open('driving_log.csv') as csvfile:
        for center_image,_,_,steering_angle,_,_, speed in csv.reader(csvfile):
            if counter > 6849:
                break
            counter += 1
            if float(steering_angle)**2 > 0.02 or float(steering_angle)**2 + .02 > random.random():
                X.append(preprocess_image(cv2.imread(center_image)))
                y.append(float(steering_angle))
                if float(steering_angle)**2 > .02:
                    X.append(preprocess_image(cv2.imread(center_image), True))
                    y.append(-float(steering_angle))

    return np.asarray(X),np.asarray(y)

X, y = generate_samples()

model = Sequential([
    Conv2D(16, strides=3, kernel_size=5, input_shape=(50, 160, 1), kernel_regularizer=regularizers.l2(.001)),
    # Conv2D(36, strides=2, kernel_size=5, kernel_regularizer=regularizers.l2(.001)),
    # Dropout(.2),
    Conv2D(32, strides=3, kernel_size=3, kernel_regularizer=regularizers.l2(.001)),
    # Dropout(.2),
    # Conv2D(32, strides=1, kernel_size=5, kernel_regularizer=regularizers.l2(.001)),
    # Dropout(.2),
    Conv2D(64, strides=2, kernel_size=2, kernel_regularizer=regularizers.l2(.001)),
    # Dropout(.2),
    Flatten(),
    # Dense(1024, activation='elu', kernel_regularizer=regularizers.l2(.001)),
    # Dropout(.5),
    Dense(512, activation='elu', kernel_regularizer=regularizers.l2(.001)),
    # Dropout(.5),
    Dense(256, activation='elu', kernel_regularizer=regularizers.l2(.001)),
    # Dropout(.5),
    Dense(128, activation='elu', kernel_regularizer=regularizers.l2(.001)),
    # Dropout(.5),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer=Adam())
history = model.fit(X,y,batch_size=32,epochs=epochs,callbacks=[csv_logger], validation_split=0.1, verbose=2)

plt.plot(range(epochs),history.history['loss'], label='Training loss' )
plt.plot( range(epochs), history.history['val_loss'], label='Validation loss')
plt.axis([0,epochs,0,max([max(history.history['loss']), max(history.history['val_loss'])])])
plt.xlabel('Epoch')
plt.legend()
plt.show()
model.save('model.h5')
