#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
LICENSE PLATE LOCALIZATION 
Created by: Udacity (c)
Modified by: Maryam Hanna
Date: July 1, 2019
Email: Maryam Hanna
MIT License, Permission is granted by Udacity(c) to obtain copy of this software. License could be found at LICENSE file. 
This algorithm detects the location of the license plate in the image, and boxes the location. When detection main is called, it returns the image and list of boxes detected.
'''


# In[ ]:


# importing important libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf 
import random
from scipy.ndimage.measurements import label


# In[ ]:


'''
locate function
Will apply sliding window with the model to predict the location of the license plate, with the help of heatmap to try to cancel out any false predictions. 
'''
def locate(model, image_read, 
           x_range = [None, None],
           y_range = [None, None], 
           pix_range = [None, None], 
           xy_overlap = (0.3, 0.3)):
    '''
    slide_window function
    This function takes the read image, and the user has the option to set the x range, y range, pix range, and the xy overlap range. 
    If the user does not set these certerias, the algorithm is set to set it up by itself
    '''
    def slide_window(image, 
                    x_range = [None, None],
                    y_range = [None, None], 
                    pix_range = [None, None], 
                    xy_overlap = (0.3, 0.3)):

        copy_image = image.copy()
        # setting up x, y, and pix range if user doesn't.
        if x_range[0] == None:
            x_range[0] = 0
        if x_range[1] == None: 
            x_range[1] = image.shape[1]
        if y_range[0] == None:
            y_range[0] = 0
        if y_range[1] == None:
            y_range[1] = image.shape[0]
        if pix_range[0] == None:
            pix_range[0] = 1
        if pix_range[1] == None:
            if (image.shape[0]*2 > image.shape[1]):
                pix_range[1] = int(image.shape[1]/2)
            else:
                pix_range[1] = image.shape[0]
        windows = []
        # sliding window going through the image, and testing each sub-section with neural network. If prediction is positive for license plate, that sub-section of image's coordinates will be saved in windows and returned at the end of the function
        for pix in range(pix_range[1], pix_range[0], -1):
            for y0 in range(y_range[0], (y_range[1]-pix), 
                            round(pix*xy_overlap[1])):
                y1 = y0 + pix
                for x0 in range(x_range[0], x_range[1]-(pix*2), 
                                round(pix*2*xy_overlap[0])):
                    x1 = x0 + pix*2
                    img = image[y0:y1, x0:x1]
                    img = cv2.resize(img, (140,70))
                    img = np.array(img).reshape(-1, 140, 70, 3)
                    pred_plate = model.predict(img)
                    if pred_plate == 1:
                        cv2.rectangle(copy_image, (x0, y0), (x1, y1), 
                                      (0,0,255), 2)
                        windows.append([(x0, y0), (x1,y1)])
        plt.imshow(copy_image)
        cv2.imwrite('boxes.JPG', copy_image)
        return windows
    '''
    heatmap creator function
    This function takes in empty image, and windows that were detected positive for license plate and returns a heatmap of most common locations where a license plate was detected. 
    '''
    def heatmap_creator(heatmap, windows):
        for box in windows:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]]+=1
        return heatmap
    '''
    apply threshold function
    This function applys threshold to the heatmap; since the location of the license plate will have more positive detectios while false positives wouldn't; hence limiting false positives.
    '''
    def apply_threshold(heatmap, threshold):
        heatmap[heatmap <= threshold] = 5
        return heatmap
    '''
    draw final box function 
    This algorithm draws a box around the heated aread in the heatmap image. 
    It returns the final image, and an array with the boxes location detected. 
    '''
    def draw_final_box(image, labels):
        boxes = []
        for plate_num in range(1, labels[1]+1):
            nonzero = (labels[0] == plate_num).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(image, box[0], box[1], (0,0,225), 3)
            boxes.append((box[0],box[1]))
        return image, boxes
    # reads in the model used to detect the plates
    model = tf.keras.models.load_model(model)
    model.summary()
    # reads in the raw images
    image = cv2.imread(image_read)
    plt.imshow(image)
    # calling on slide_window function to get the list of windows
    windows = slide_window(image, x_range=[None, None],
                          y_range = y_range, 
                          pix_range = pix_range, 
                          xy_overlap = (0.3, 0.3))
    # creating heat image, and send through the heatmap_creator function with windows
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = heatmap_creator(heat, windows)
    plt.imshow(heat)
    # applying threshold function is called 
    heat = apply_threshold(heat, 30)
    plt.imshow(heat)
    heatmap = np.clip(heat, 0, 255)
    heat_image=heatmap
    # find final boxes from heatmap using label function from scipy.ndimage.measurements
    labels = label(heatmap)
    # print the number of boxes detected and the image with boxes
    print(" Number of Plates found - ",labels[1])
    draw_img, boxes = draw_final_box(np.copy(image), labels)
    plt.imshow(draw_img)
    return labels
# END OF CODE

