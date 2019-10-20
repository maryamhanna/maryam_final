#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
FINAL NEURAL NETWORK CODE
Created by: Udacity (C)
Modified by: Maryam Hanna
Date: June 1, 2019
Email: maryamhanna@hotmail.com
MIT License, Permission is granted by Udacity(c) to obtain copy of this software. License could be found at LICENSE file. 
This algorithm trains a neural network to accurately detect license plates. This code requires 'data.csv' which should contain the file's location for training the neural network. The current setup algorithm trains 108 different neural networks. The user could modify the  differnt types of dropout levels, the number of dense layer, the number of convolution layer, and the size of the hidden layer. 
If for whatever reason, the neural network crashes, the user does not have to restart the whole training process, they could count the number the neural network is at and modify count at variable. 
'''


# In[ ]:


#importing required libraries
import csv
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from sklearn.utils import shuffle
from keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPooling2D, Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
import keras.optimizers


# In[ ]:


'''
train_neural_network function
This function trains a neural network with the given data and conditions, and then saves the model and clears the computer’s memory so it does not crash.  
'''
def train_neural_network(data_loc, 
                         dropout_precent, 
                         dense_,
                         conv_,
                         size_, 
                         batch__, 
                         learning_rate, 
                         learning_decay, 
                         epochs_, 
                         savemodel_loc): 
# reading in the labeled data from data.csv
data = []
with open(data_loc) as file:
    reader = csv.reader(file)
    for line in reader:
        data.append(line)
# splitting the read data into training, and validation sets
train_data, valid_data = train_test_split(data, test_size=0.25)
'''
generator function
When this function is called, it reads in the input lines of the data and its
labels, along with the batch size. It returns an array of the images, and their measurements.
Its a yield return type of function: meaning, it doesn't lose track of where it is in the input lines.
'''
def generator(input_lines, batch_size):
    num_samples = len(input_lines)
    while 1:
        shuffle(input_lines)
        for offset in range(0, num_samples, batch_size):
            batch_lines = input_lines[offset:offset+batch_size]
            images = []
            measurements = []
            for each_batch_line in batch_lines:
                current_path=each_batch_line[0]
                image=cv2.imread(current_path)
                # image resized and brought to binary neural network
                image = cv2.resize(image, (140,70))
                image = image/255.0
                images.append(image)
                measurement=each_batch_line[1]
                measurements.append(measurement)
            images = np.array(images).reshape(-1, 140, 70, 3)  
            yield np.array(images), np.array(measurements)
# compile and train the model using generator function, 
batch_size= batch__size
train_gen = generator(train_data, batch_size)
valid_gen = generator(valid_data, batch_size)
# activating access to GPU -- comment out if using CPU only
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))
# using GPU percentage 75%; modify as need be
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# setting up the different types of neural networks to be constructed
count = 0
dropouts = dropout_precent
dense_layers = dense_
conv_layers = conv_
layer_sizes = size_
for dropout in dropouts:
    for dense in dense_layers:
        for conv in conv_layers:
            for size in layer_sizes:
                if count => 0:
                    NAME = 'drop' + str(int(dropout*100)) + '_dense' + 
                            str(dense) + '_conv' + str(conv)+ '_size' 
                            +str(size)
                    print(NAME)
                    # start of neural network
                    model = Sequential()
                    # input layer
                    model.add(Conv2D(size, (1,1), 
                                     input_shape=(140,70,3))) 
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2,2)))
                    model.add(Activation('relu'))
                    model.add(Dropout(dropout))
                    # convolution layers
                    for l in range(conv-1):
                        model.add(Conv2D(size, (1,1)))
                        model.add(Activation('relu'))
                        model.add(MaxPooling2D(pool_size=(2,2)))
                        model.add(Activation('relu'))
                        model.add(Dropout(dropout))
                    model.add(Flatten())
                    # dense layers -- set to all have 128 size
                    for _ in range(dense):
                        model.add(Dense(128))                                               
                        model.add(Activation('relu'))  
                        model.add(Dropout(dropout))
                    # sigmoid operation for binary output
                    model.add(Dense(1))
                    model.add(Activation('sigmoid'))
                    # adam optimizer is decaying learning rate network
                    adam = Adam(lr=learning_rate, decay=learning_decay)
                    # binary-crossentropy is used validate each epoch
                    model.compile(optimizer=adam,
                                  loss='binary_crossentropy',
                                  metrics=['accuracy'])
                    steps_per_epoch_=len(train_data)
                    checkpointer = ModelCheckpoint(filepath=
                                                  "{}.hdf5".format(NAME), 
                                                   verbose=1, 
                                                   save_best_only=True)
                    tensorboard = TensorBoard(log_dir=
                                                  "logs/{}".format(NAME))
                                  history=model.fit_generator(train_gen, 
                                  steps_per_epoch=steps_per_epoch_, 
                                  validation_data=valid_gen, 
                                  validation_steps=len(valid_data), 
                                  verbose=1, epochs=epochs_, 
                                  callbacks=[tensorboard])
                    # save model then clear memory to training next model
                    model.save("{}{}.model".format(savemodel_loc,NAME))
                    print("saved model")
                    del model
                    del history
                else:
                    count += 1
# END OF CODE

