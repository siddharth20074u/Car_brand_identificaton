#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:43:00 2020

@author: siddharthsmac
"""

from Resnet_50 import test_set
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

model=load_model('model_resnet50_car_brand.h5')

y_pred = model.predict(test_set)

y_pred = np.argmax(y_pred, axis=1)

img=image.load_img('Datasets/Test/lamborghini/11.jpg',target_size=(224,224))

x=image.img_to_array(img)

x=x/255

x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)

model.predict(img_data)

a=np.argmax(model.predict(img_data), axis=1)

print(a)
