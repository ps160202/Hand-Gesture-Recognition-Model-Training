import os

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import load_model

model = load_model('handrecognition_model.h5')

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if path.endswith("jpg") and path.startswith(".\\imgL"):
            testImagePath = path

print(testImagePath)

class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]

img = cv2.imread(testImagePath)

X=[]

if (img is not None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts into the corret colorspace (GRAY)
    img = cv2.resize(img, (320, 120))  # Reduce image size so training can be faster
    X.append(img)

X = np.array(X, dtype="uint8")
X = X.reshape(1, 120, 320, 1)

prediction = model.predict(X)

predicted_label = np.argmax(prediction)


plt.imshow(img, cmap=plt.cm.binary)
print(class_names[predicted_label])
print(100 * np.max(prediction))

plt.show()
