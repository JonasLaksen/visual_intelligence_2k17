from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Lambda
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import time
from keras.callbacks import CSVLogger
import csv
import cv2
import numpy as np
csv_logger = CSVLogger(time.strftime("./logs/%m-%d-%H:%M:%S"), append=True, separator=';')

def preprocess_image(x):
    x=cv2.imread(x,0)
    x=cv2.resize(x, (64,32))
    x=x.reshape((32,64,1))
    return x/255.0

def generate_samples():
    X,y = [],[]
    with open('driving_log.csv') as csvfile:
        for center_image,_,_,steering_angle,_,_, speed in csv.reader(csvfile):
            if float(speed) < 20: continue
            # if np.random.uniform() < .5 and float(steering_angle) == 0.0: continue
            X.append(preprocess_image(center_image))
            y.append(float(steering_angle))

    return np.asarray(X),np.asarray(y)

X, y = generate_samples()

model = Sequential([
    Conv2D(32, strides=3, kernel_size=3, input_shape=(32, 64, 1), name='wtf'),
    Flatten(),
    Dense(256, activation='elu'),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer=Adam())
history = model.fit(X,y,batch_size=32,epochs=50,callbacks=[csv_logger], validation_split=0.1, verbose=2)
model.save('model.h5')
