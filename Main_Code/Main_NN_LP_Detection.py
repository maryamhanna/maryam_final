#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
MAIN NEURAL NETWORK LICNESE PLATE DETECTION
Created by: Maryam Hanna
July 3, 2019
Email: maryamhanna@hotmail.com
This algorithm does what the user wants, which is create a data cvs, train a neural network, 
test a neural network, find the location of the license plate in raw image using a model, or 
detecting the distance of the detected license plate. 
'''


# In[ ]:


# importing required libraries
import csv
import numpy as np
import random
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
from keras import backend as K
import tensorflow as tf
from keras.layers import Sequential, Flatten, Dense, MaxPooling2D, Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
import keras.optimizers
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
# importing other files
from CSV_Maker import *
from Final_Neural_Network import *
from Neural_Network_Tester import *
from License_Plate_Localization import *
from License_Plate_Distance_Detection import *


# In[ ]:


# requirements to call on creating csv function
create_csv = True
dir_loc = "C:/Users/wicip/Desktop/"
not_plates_file = "no_plate"
plates_file = "plates"
savefile = "data_test.csv"
# requirements to call on training neural network function
train_nn = True
data_loc = "C:/Users/wicip/Documents/working_nn/final/data.csv"
dropout_precent = 0.1
dense_layers = 1 
conv_layers = 3 
layer_size = 64  
batch_size = 8 
learning_rate = 0.00001 
learning_decay = 1e-6
epochs = 10
savemodel_loc = "C:/Users/wicip/Documents/working_nn/final/"


# In[ ]:


# requirements to call on neural network tester function
test_nn = True
test_data = "C:/Users/wicip/Documents/working_nn/final/data_test.csv"
test_model = "C:/Users/wicip/Documents/working_nn/final/drop20_dense1_con4_size64_FINAL.model"
# requirements to call on license plate localization function
localization = True
image = "C:/Users/wicip/Documents/working_nn/final/test1.JPG"
model = "C:/Users/wicip/Documents/working_nn/final/3-128-2-1553464259.model"
x_range = [None, None]
y_range = [250, 360]
pix_range = [7, 95]
xy_overlap = (0.3, 0.3)


# In[ ]:


# there are no requirements to call on distance detection function, but user must run localization function as well, since distance detection function depends on the localization function
distance_detection = True
## DO NOT EDIT BELOW HERE ##
if create_csv:
    create_csv(dir_loc, not_plates_file, plates_file, savefile)
if train_nn:
    train_neural_network(data_loc, 
                         dropout_precent, 
                         dense_layers,
                         conv_layers,
                         layer_size, 
                         batch_size, 
                         learning_rate, 
                         learning_decay, 
                         epochs, 
                         savemodel_loc)
if test_nn:
    test_neural_network(test_data, test_model) 
if localization:
    labels = locate(model, image, 
           x_range_ = x_range,
          y_range_ = y_range,
          pix_range_ = pix_range, 
          xy_overlap_ = xy_overlap)   
if localization and distance_detection:
    dist_detect(image, labels)
elif distance_detection and not localization:
    print("Must construct localization algorithm to run distance_detection algorithm")
# END CODE 

