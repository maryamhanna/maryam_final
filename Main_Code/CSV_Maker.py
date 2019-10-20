#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
CSV MAKER CODE 

Created by: Maryam Hanna
Date: April 14, 2019
Email: maryamhanna@hotmail.com

This code is used to load large data-sets into a cvs file, to help avoid over-loading the computer when training a neural network,
'''


# In[2]:


# importing required libraries 
import csv
import numpy as np
import random
import os
from PIL import Image
from sklearn.model_selection import train_test_split


# In[3]:


#labelling the categories, which is also the files' name of where the images are loacted; datadir is the computer pathways
CATEGORIES = ["cars_no_plate/", "cars0_plate/"]
DATADIR = "C:/Users/wicip/Desktop/"
files = []
#loads every image in each category and appends it into files array, along with the image's label
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for filename in (os.listdir(path)):
        try:
            filepath = os.path.join(path,filename)
            img = Image.open(filepath)
            files.append([filepath, class_num])
        except IOError:
            print("error in reading:", filename)
            pass
#shuffeling the files, so all the neural network needs to do is load the images
random.shuffle(files)
print(len(files))
#train_lines, validation_lines = train_test_split(files, test_size=0.3)
#print(len(train_lines))
#print(len(validation_lines))


# In[4]:


#opening data.csv file, and loading the files' data

with open('data_test.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(files)
writeFile.close()


# In[4]:


TESTDIR = "C:/Users/wicip/Documents/maryam_final/Images/testSetSmall/"
files = []
for filename in (os.listdir(TESTDIR)):
    try: 
        filepath = os.path.join(TESTDIR, filename)
        #print(filepath)
        img = Image.open(filepath)
        files.append([filepath])
    except IOError:
        print("error in reading:", filename)
        pass
random.shuffle(files)


# In[5]:


with open('test_data.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(files)
writeFile.close()


# In[ ]:




