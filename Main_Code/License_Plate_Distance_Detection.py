#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
LICENSE PLATE DISTANCE DETECTOR 
Created by: Maryam Hanna
June 20, 2019
Email: maryamhann@hotmail.com
This algorithm produces the distance of the detected license plates.
'''


# In[ ]:


#importing required libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf 
import random


# In[ ]:


'''
dist_detect function 
Determines the distance of the vehicle through the size of the license plate. Then prints out the results on the image. 
'''
def dist_detect(image_read, labels):
'''
plate size function 
Determines the distance of the vehicle, according to the size of the plate detecte in the image, and returns the calculated distance
'''
def plate_size(x_pixels, y_pixels):
    ### x_distance 
    if x_pixels >= 325:
        x_distance = 0.0
    elif (325 > x_pixels > 307):
        x_distance = 1.0
    elif (307 >= x_pixels >= 161):
        x_distance = 1.0 + x_pixels/(305+165)
    elif (161 > x_pixels > 157):
        x_distance = 2.0
    elif (157 >= x_pixels >= 108):
        x_distance = 2.0 + x_pixels/(155+110)
    elif (108 > x_pixels > 104):
        x_distance = 3.0
    elif(104 >= x_pixels >= 82):
        x_distance = 3.0 + x_pixels/(100+82)
    elif (82 > x_pixels > 79):
        x_distance = 4.0
    elif (79 >= x_pixels >= 66):
        x_distance = 4.0 + x_pixels/(79+66) 
    elif (66 > x_pixels > 63):
        x_distance = 5.0
    elif (63 >= x_pixels >= 57):
        x_distance = 5.0 + x_pixels/(63+57)
    elif (57 > x_pixels > 53):
        x_distance = 6.0
    elif (53 >= x_pixels >= 48):
        x_distance = 6.0 + x_pixels/(53+48)
    elif (48 > x_pixels > 45):
        x_distance = 7.0
    elif (45 >= x_pixels > 41):
        x_distance = 7.0 + x_pixels/(45+42)
    elif (x_pixels == 41):
        x_distance = 8.0
    elif (41 > x_pixels >= 38):
        x_distance = 8.0 + x_pixels/(42+38)
    elif (38 > x_pixels > 35):
        x_distance = 9.0
    elif (35 >= x_pixels >= 34):
        x_distance = 9.0 + x_pixels/(35+34)
    elif (34 > x_pixels > 31):
        x_distance = 10.0
    elif (x_pixels == 31):
        x_distance = 10.5
    elif (x_pixels == 30):
        x_distance = 11.0
    elif (x_pixels == 29):
        x_distance = 11.5
    elif (29 > x_pixels > 26):
        x_distance = 12.0
    elif (x_pixels == 26):
        x_distance = 12.5
    elif (x_pixels == 25):
        x_distance = 13.0
    elif (x_pixels == 24):
        x_distance = 14.0
    elif (x_pixels == 23):
        x_distance = 14.5
    elif (x_pixels == 22):
        x_distance = 15.0
    elif (x_pixels == 21):
        x_distance = 16.0
    elif (x_pixels == 20):
        x_distance = 17.0
    elif (x_pixels == 19):
        x_distance = 18.0
    elif (x_pixels == 18):
        x_distance = 19.0
    elif (x_pixels == 17):
        x_distance = 20.0
        ## skipping distance 21 
    elif (x_pixels == 16):
        x_distance = 22.0
        ## skipping distance 23 
    elif (x_pixels == 15):
        x_distance = 24.0
        ## skipping distance 25
    elif (x_pixels == 14):
        x_distance = 26.0
        ## skipping distance 27 
    elif (x_pixels == 13):
        x_distance = 28.0
    elif (x_pixels == 12):
        x_distance = 29.0
    elif (x_pixels < 12):
        x_distance = 30.0
    ### y_distance 
    if (y_pixels >= 172):
        y_distance = 0.0
    elif (172 > y_pixels > 161):
        y_distance = 1.0
    elif (161 >= y_pixels >= 87):
        y_distance = 1.0 + y_pixels/(161+87)
    elif (87 > y_pixels > 82):
        y_distance = 2.0
    elif (82 >= y_pixels >= 59):
        y_distance = 2.0 + y_pixels/(82+59)
    elif (59 > y_pixels > 55):
        y_distance = 3.0
    elif (55 >= y_pixels >=45):
        y_distance = 3.0 + y_pixels/(55+45)
    elif (45 > y_pixels > 42):
        y_distance = 4.0
    elif (42 >= y_pixels >= 37):
        y_distance = 4.0 + y_pixels/(42+37)
    elif (37 > y_pixels > 34):
        y_distance = 5.0
    elif (34 >= y_pixels >= 32):
        y_distance = 5.0 + y_pixels/(34+32)
    elif (32 > y_pixels > 28):
        y_distance = 6.0
    elif (28 >= y_pixels > 26):
        y_distance = 6.0 + y_pixels/(28+27)
    elif (y_pixels == 26):
        y_distance = 7.0
    elif (y_pixels == 25):
        y_distance = 7.5
    elif (25 > y_pixels > 22):
        y_distance = 8.0
    elif (y_pixels == 22):
        y_distance = 8.5
    elif (22 > y_pixels > 19):
        y_distance = 9.0
    elif (19 >= y_pixels > 17):
        y_distance = 10.0
    elif (y_pixels == 17):
        y_distance = 11.0
    elif (y_pixels == 16):
        y_distance = 12.0
    elif (y_pixels == 15):
        y_distance = 13.0
    elif (y_pixels == 14):
        y_distance = 14.0
    elif (y_pixels == 13):
        y_distance = 15.0
    elif (y_pixels == 12):
        y_distance = 16.0
        ## skipping distance 17 because it has 12 pixel 
    elif (y_pixels == 11):
        y_distance = 18.0
    elif (y_pixels == 10):
        y_distance = 19.0
        ## skipping distance 20 because it has 10 pixel 
    elif (y_pixels == 9):
        y_distance = 21.0
        ## skipping distances 22, 23, 24, 25 
    elif (y_pixels == 8):
        y_distance = 26.0
        ## skipping distances 27, 28, 29 because they all have 8 pixel 
    elif (y_pixels == 7):
        y_distance = 30.0
    elif (y_pixels < 7):
        y_distance = 30.5
        
    if (y_distance == 16.0):
        distance = x_distance 
    elif (y_distance == 19.0):
        distance = x_distance 
    elif (y_distance == 21.0):
        distance = x_distance 
    elif (y_distance == 26.0):
        distance = x_distance 
    elif (x_distance == 20.0):
        distance = y_distance 
    else:
        distance = (x_distance+y_distance)/2
    print(x_distance)
    print(y_distance)
    print(distance)
    return distance
# read given image       
image = cv2.imread(image_read)
plt.imshow(image)
# boxes on image and write distance predicted next to it
final_img = np.copy(image_read)
for plate in range(labels[1]):
    boxes[plate]
    x_pixels = boxes[plate][1][0]-boxes[plate][0][0]
    y_pixels = boxes[plate][1][1]-boxes[plate][0][1]
    distance = plate_size(x_pixels, y_pixels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    distance_str = str(round(distance,2))
    distance_str = distance_str + ' m'
    final_img = cv2.putText(final_img, distance_str, (boxes[plate][1][0], boxes[plate][0][1]), font, 2,(0,255,0), 2, cv2.LINE_AA)
# print the final image
plt.imshow(final_img)
# END OF CODE

