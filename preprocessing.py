from cv2 import cv2


def preprocess_image(x, flip=False):
    # x=cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    if flip:
        x = cv2.flip(x,1)
    x=x[50:150,0:320].copy()
    x=cv2.resize(x, (160,50))
    x=x.reshape((50,160,3))
    return x/255.0

