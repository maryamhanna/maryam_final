#!/usr/bin/env python
# coding: utf-8

'''
CALIBRATE RAW IMAGES CODE

Created by: Maryam Hanna
Date: November 21, 2018
Email: maryamhanna@hotmail.com

This is image calbration code specific for Nikon 1 J1 raw images. The resolution of time images used is 1280x720. The chessboard used for the camera matrix 9x7.
This code uses pickle file which has the camera matrix.  
'''

# importing required libraries
import numpy as np 
import pickle
import cv2
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().run_line_magic('matplotlib', 'inline')

# load pickle file with camera matrix
pickle_in = open("C:/Users/wicip/Documents/maryam_final/Codes/Calibration/dist_pickle.p", "rb")
array = pickle.load(pickle_in)
mtx = array['mtx']
dist = array['dist']
pickle_in.close()

# load the raw images that will be calibrated
mypath = "C:/Users/wicip/Documents/maryam_final/Images/Raw_Images/"
dirs = os.listdir(mypath)
files = []
for i in range (len(dirs)):
    if "Car" in dirs[i]:
        files.append(dirs[i])

# for loop goes through every image to calibrate it, and then save it
save_path = "C:/Users/wicip/Documents/maryam_final/Images/Camera_Calibration/cal_cars"
for i in range(len(files)):
    new_path = os.path.join(mypath, files[i])
    for filename in os.listdir(new_path):
        location = new_path + '/' + filename
        # read in image
        img = cv2.imread(location)
        if img is not None:
            # undistort image
            undist_img = cv2.undistort(img, mtx, dist, None, mtx)
            # save image 
            img_name = save_path+'/cal_'+filename
            cv2.imwrite(img_name, undist_img)

# END OF CODE 