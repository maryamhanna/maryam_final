#!/usr/bin/env python
# coding: utf-8

# In[3]:


'''
NEURAL NETWORK TESTER
Created by: Maryam Hanna
Date: June 3, 2019
Email: maryamhanna@hotmail.com
This algorithm tests out the neural network and prints out the results to the user. 
'''


# In[ ]:


# importing important libraries
import tensorflow as tf
import numpy as np
import cv2
import csv
import os


# In[4]:


'''
test_neural_network function
Reads in the test data and the test model, and after predicting the test data, it compares it results and prints out accuracy.
'''
def test_neural_network(test_data, test_model): 
    # reading in the labeled data from data_test.csv
    data = []
    with open(test_data) as file:
        reader = csv.reader(file)
        for line in reader:
            data.append(line)
    # splitting the images from their labels
    test_images = []
    test_labels = []
    for i in range(len(data)):
        test_images.append(data[i][0])
        test_labels.append(data[i][1])
    # reading in the model that we want to test
        print(test_model)
        model = tf.keras.models.load_model(test_model)

    # predict each read image and test accuracy
    accuracy = 0
    for i in range(len(test_images)):
        img = cv2.imread(test_images[i])
        img = cv2.resize(img, (140,70))
        img = np.array(img).reshape(-1, 140, 70, 3)
        pred_plate = model.predict(img)
        pred_num = str(int(pred_plate))
        if round(pred_num) == test_labels[i]:
            accuracy += 1
    # prints the accuracy to the user   
    print('Number of tested images = ', len(test_images))
    print('Number of correct prediction = ', accuracy)
    print('Number of wrong prediction = ', len(test_images)-accuracy)
    print('Successful percentage = ', accuracy/len(test_images), '%')
    #END OF CODE 

