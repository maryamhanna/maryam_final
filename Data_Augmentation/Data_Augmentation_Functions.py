#!/usr/bin/env python
# coding: utf-8

'''
DATA AUGMENTATION FUNCTIONS CODE

Created by: Maryam Hanna
Date: December 20, 2018
Email: maryamhanna@hotmail.com

This is data augmentation code written for Data_Augmentation_Main.py code. It has five different functions: test_data, flipping_image, rotating_image, salt_pepper_image, and saturate_contrast_image.

References:
1) https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
2) https://stackoverflow.com/questions/42045362/change-contrast-of-image-in-pil
3) https://docs.opencv.org/3.4.2/d3/dc1/tutorial_basic_linear_transform.html
'''

# importing required libraries
import cv2 
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

'''
test_data function

When this function is called, it reads in a test image and operates on it different types of augmentation, and then prints them together for comparision. It compares flipped image, rotating image, added salt and pepper noise to image, and modifies contrast and brightness of the image. It is recommended that the user has 'test1.JPG' image in test_load folder and have test_save folder available to save the images to.
'''

def test_data():
    
    # pathway to test image and reading original image
    testload = 'C:/Users/wicip/Documents/transfer-master/test_load/'
    testsave = 'C:/Users/wicip/Documents/transfer-master/test_save/'
    org_img = cv2.imread(os.path.join(testload, 'test.JPG'))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    # calling on the flipping_image function to flip the main image
    flipping_image(testload, testsave)
    
    # calling on the rotating_image function to rotate the main image at angles 15, 30, 45, 60, 75 degrees 
    rotating_image(testload, testsave, 15)
    rotating_image(testload, testsave, 30)
    rotating_image(testload, testsave, 45)
    rotating_image(testload, testsave, 60)
    rotating_image(testload, testsave, 75)
    
    # calling on the salt_pepper_image function to add salt and pepper noise to the main image to comparison between ratio of salt and pepper of 25%, 50%, 75% and the amount of salt of pepper, of 10%, 8%, 6%, 4%, 2%
    salt_pepper_image(testload, testsave, 0.25, 0.1)
    salt_pepper_image(testload, testsave, 0.5, 0.1)
    salt_pepper_image(testload, testsave, 0.75, 0.1)
    salt_pepper_image(testload, testsave, 0.25, 0.08)
    salt_pepper_image(testload, testsave, 0.5, 0.08)
    salt_pepper_image(testload, testsave, 0.75, 0.08)
    salt_pepper_image(testload, testsave, 0.25, 0.06)
    salt_pepper_image(testload, testsave, 0.5, 0.06)
    salt_pepper_image(testload, testsave, 0.75, 0.06)
    salt_pepper_image(testload, testsave, 0.25, 0.04)
    salt_pepper_image(testload, testsave, 0.5, 0.04)
    salt_pepper_image(testload, testsave, 0.75, 0.04)
    salt_pepper_image(testload, testsave, 0.25, 0.02)
    salt_pepper_image(testload, testsave, 0.5, 0.02)
    salt_pepper_image(testload, testsave, 0.75, 0.02)
    
    # calling on the saturate_contrast_image function to modify the main image's contrast and saturation at alpha (contrast which ranges from 1 to 3) being at 1, 1.5, 2, 2.5, 3 and beta (brightness which ranges from 0 to 100) being at 0, 25, 50, 75, 100
    saturate_contrast_image(testload, testsave, 1, 0)
    saturate_contrast_image(testload, testsave, 1, 25)
    saturate_contrast_image(testload, testsave, 1, 50)
    saturate_contrast_image(testload, testsave, 1, 75)
    saturate_contrast_image(testload, testsave, 1, 100)
    saturate_contrast_image(testload, testsave, 1.5, 0)
    saturate_contrast_image(testload, testsave, 1.5, 25)
    saturate_contrast_image(testload, testsave, 1.5, 50)
    saturate_contrast_image(testload, testsave, 1.5, 75)
    saturate_contrast_image(testload, testsave, 1.5, 100)
    saturate_contrast_image(testload, testsave, 2, 0)
    saturate_contrast_image(testload, testsave, 2, 25)
    saturate_contrast_image(testload, testsave, 2, 50)
    saturate_contrast_image(testload, testsave, 2, 75)
    saturate_contrast_image(testload, testsave, 2, 100)
    saturate_contrast_image(testload, testsave, 2.5, 0)
    saturate_contrast_image(testload, testsave, 2.5, 25)
    saturate_contrast_image(testload, testsave, 2.5, 50)
    saturate_contrast_image(testload, testsave, 2.5, 75)
    saturate_contrast_image(testload, testsave, 2.5, 100)
    saturate_contrast_image(testload, testsave, 3, 0)
    saturate_contrast_image(testload, testsave, 3, 25)
    saturate_contrast_image(testload, testsave, 3, 50)
    saturate_contrast_image(testload, testsave, 3, 75)
    saturate_contrast_image(testload, testsave, 3, 100)
    
    # flip_fig and flip_ax are used to set up plot for multiple images, with flip_fig used for the figures and flip_ax used to label the images
    print("\n\nFlipped Image Comparision\n")
    flip_fig = plt.figure()
    flip_ax = []
    # reads in flipped image
    flip_img = cv2.imread(os.path.join(testsave, 'flip_test.JPG'))
    flip_img = cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)
    # plotting the images and their respected titles
    flip_ax.append(flip_fig.add_subplot(2, 1, 1))
    flip_ax[-1].set_title("original")
    plt.imshow(org_img)
    flip_ax.append(flip_fig.add_subplot(2, 1, 2))
    flip_ax[-1].set_title("flipped")
    plt.imshow(flip_img)
    flip_fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()
      
    # rot_fig and rot_ax are used to set up plot for multiple images, with rot_fig used for the figures and rot_ax used to label the images
    print("\n\nRotated Images Comparision\n")
    rot_fig = plt.figure()
    rot_ax = []
    # reads in rotated images
    rot15_img = cv2.imread(os.path.join(testsave, 'rot15_test.JPG')) 
    rot15_img = cv2.cvtColor(rot15_img, cv2.COLOR_BGR2RGB)
    rot30_img = cv2.imread(os.path.join(testsave, 'rot30_test.JPG')) 
    rot30_img = cv2.cvtColor(rot30_img, cv2.COLOR_BGR2RGB)
    rot45_img = cv2.imread(os.path.join(testsave, 'rot45_test.JPG')) 
    rot45_img = cv2.cvtColor(rot45_img, cv2.COLOR_BGR2RGB)
    rot60_img = cv2.imread(os.path.join(testsave, 'rot60_test.JPG'))
    rot60_img = cv2.cvtColor(rot60_img, cv2.COLOR_BGR2RGB)
    rot75_img = cv2.imread(os.path.join(testsave, 'rot75_test.JPG')) 
    rot75_img = cv2.cvtColor(rot75_img, cv2.COLOR_BGR2RGB)
    # plotting the images and their respected titles
    rot_ax.append(rot_fig.add_subplot(3, 2, 1))
    rot_ax[-1].set_title("original")
    plt.imshow(org_img)
    rot_ax.append(rot_fig.add_subplot(3, 2, 2))
    rot_ax[-1].set_title("rot15")
    plt.imshow(rot15_img)
    rot_ax.append(rot_fig.add_subplot(3, 2, 3))
    rot_ax[-1].set_title("rot30")
    plt.imshow(rot30_img)
    rot_ax.append(rot_fig.add_subplot(3, 2, 4))
    rot_ax[-1].set_title("rot45")
    plt.imshow(rot45_img)
    rot_ax.append(rot_fig.add_subplot(3, 2, 5))
    rot_ax[-1].set_title("rot60")
    plt.imshow(rot60_img)
    rot_ax.append(rot_fig.add_subplot(3, 2, 6))
    rot_ax[-1].set_title("rot75")
    plt.imshow(rot75_img)
    rot_fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()
    
    # sp_fig and sp_ax are used to set up plot for multiple images, with sp_fig used for the figures and sp_ax used to label the images
    print("\n\nSalt and Peppered Images Comparision\n")
    sp_fig = plt.figure()
    sp_ax = []
    # reads in salt and peppered images
    sp_r25_a1_img = cv2.imread(os.path.join(testsave, 'sp_r25_a10_test.JPG'))
    sp_r25_a1_img = cv2.cvtColor(sp_r25_a1_img, cv2.COLOR_BGR2RGB)
    sp_r50_a1_img = cv2.imread(os.path.join(testsave, 'sp_r50_a10_test.JPG'))
    sp_r50_a1_img = cv2.cvtColor(sp_r50_a1_img, cv2.COLOR_BGR2RGB)
    sp_r75_a1_img = cv2.imread(os.path.join(testsave, 'sp_r75_a10_test.JPG'))
    sp_r75_a1_img = cv2.cvtColor(sp_r75_a1_img, cv2.COLOR_BGR2RGB)
    sp_r25_a08_img = cv2.imread(os.path.join(testsave, 'sp_r25_a8_test.JPG'))
    sp_r25_a08_img = cv2.cvtColor(sp_r25_a08_img, cv2.COLOR_BGR2RGB)
    sp_r50_a08_img = cv2.imread(os.path.join(testsave, 'sp_r50_a8_test.JPG'))
    sp_r50_a08_img = cv2.cvtColor(sp_r50_a08_img, cv2.COLOR_BGR2RGB)
    sp_r75_a08_img = cv2.imread(os.path.join(testsave, 'sp_r75_a8_test.JPG'))
    sp_r75_a08_img = cv2.cvtColor(sp_r75_a08_img, cv2.COLOR_BGR2RGB)
    sp_r25_a06_img = cv2.imread(os.path.join(testsave, 'sp_r25_a6_test.JPG'))
    sp_r25_a06_img = cv2.cvtColor(sp_r25_a06_img, cv2.COLOR_BGR2RGB)
    sp_r50_a06_img = cv2.imread(os.path.join(testsave, 'sp_r50_a6_test.JPG'))
    sp_r50_a06_img = cv2.cvtColor(sp_r50_a06_img, cv2.COLOR_BGR2RGB)
    sp_r75_a06_img = cv2.imread(os.path.join(testsave, 'sp_r75_a6_test.JPG'))
    sp_r75_a06_img = cv2.cvtColor(sp_r75_a06_img, cv2.COLOR_BGR2RGB)
    sp_r25_a04_img = cv2.imread(os.path.join(testsave, 'sp_r25_a4_test.JPG'))
    sp_r25_a04_img = cv2.cvtColor(sp_r25_a04_img, cv2.COLOR_BGR2RGB)
    sp_r50_a04_img = cv2.imread(os.path.join(testsave, 'sp_r50_a4_test.JPG'))
    sp_r50_a04_img = cv2.cvtColor(sp_r50_a04_img, cv2.COLOR_BGR2RGB)
    sp_r75_a04_img = cv2.imread(os.path.join(testsave, 'sp_r75_a4_test.JPG'))
    sp_r75_a04_img = cv2.cvtColor(sp_r75_a04_img, cv2.COLOR_BGR2RGB)
    sp_r25_a02_img = cv2.imread(os.path.join(testsave, 'sp_r25_a2_test.JPG'))
    sp_r25_a02_img = cv2.cvtColor(sp_r25_a02_img, cv2.COLOR_BGR2RGB)
    sp_r50_a02_img = cv2.imread(os.path.join(testsave, 'sp_r50_a2_test.JPG'))
    sp_r50_a02_img = cv2.cvtColor(sp_r50_a02_img, cv2.COLOR_BGR2RGB)
    sp_r75_a02_img = cv2.imread(os.path.join(testsave, 'sp_r75_a2_test.JPG'))
    sp_r75_a02_img = cv2.cvtColor(sp_r75_a02_img, cv2.COLOR_BGR2RGB)
    # plotting the images and their respected titles
    sp_ax.append(sp_fig.add_subplot(4, 4, 1))
    sp_ax[-1].set_title("original")
    plt.imshow(org_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 2))
    sp_ax[-1].set_title("sp_r0.25_a0.1")
    plt.imshow(sp_r25_a1_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 3))
    sp_ax[-1].set_title("sp_r0.50_a0.1")
    plt.imshow(sp_r50_a1_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 4))
    sp_ax[-1].set_title("sp_r0.75_a0.1")
    plt.imshow(sp_r75_a1_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 5))
    sp_ax[-1].set_title("sp_r0.25_a0.08")
    plt.imshow(sp_r25_a08_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 6))
    sp_ax[-1].set_title("sp_r0.50_a0.08")
    plt.imshow(sp_r50_a08_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 7))
    sp_ax[-1].set_title("sp_r0.75_a0.08")
    plt.imshow(sp_r75_a08_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 8))
    sp_ax[-1].set_title("sp_r0.25_a0.06")
    plt.imshow(sp_r25_a06_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 9))
    sp_ax[-1].set_title("sp_r0.50_a0.06")
    plt.imshow(sp_r50_a06_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 10))
    sp_ax[-1].set_title("sp_r0.75_a0.06")
    plt.imshow(sp_r75_a06_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 11))
    sp_ax[-1].set_title("sp_r0.25_a0.04")
    plt.imshow(sp_r25_a04_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 12))
    sp_ax[-1].set_title("sp_r0.50_a0.04")
    plt.imshow(sp_r50_a04_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 13))
    sp_ax[-1].set_title("sp_r0.75_a0.04")
    plt.imshow(sp_r75_a04_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 14))
    sp_ax[-1].set_title("sp_r0.25_a0.02")
    plt.imshow(sp_r25_a02_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 15))
    sp_ax[-1].set_title("sp_r0.50_a0.02")
    plt.imshow(sp_r50_a02_img)
    sp_ax.append(sp_fig.add_subplot(4, 4, 16))
    sp_ax[-1].set_title("sp_r0.75_a0.02")
    plt.imshow(sp_r75_a02_img)
    sp_fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=2.0)
    plt.show()
    # reads in contrasted and saturated images
    print("\n\nContrast and Brightening the Images Comparision\n")
    cont_a10_b0_img = cv2.imread(os.path.join(testsave, 'cont_a10_b0_test.JPG'))
    cont_a10_b0_img = cv2.cvtColor(cont_a10_b0_img, cv2.COLOR_BGR2RGB)
    cont_a10_b25_img = cv2.imread(os.path.join(testsave, 'cont_a10_b25_test.JPG')) 
    cont_a10_b25_img = cv2.cvtColor(cont_a10_b25_img, cv2.COLOR_BGR2RGB)
    cont_a10_b50_img = cv2.imread(os.path.join(testsave, 'cont_a10_b50_test.JPG'))
    cont_a10_b50_img = cv2.cvtColor(cont_a10_b50_img, cv2.COLOR_BGR2RGB)
    cont_a10_b75_img = cv2.imread(os.path.join(testsave, 'cont_a10_b75_test.JPG'))
    cont_a10_b75_img = cv2.cvtColor(cont_a10_b75_img, cv2.COLOR_BGR2RGB)
    cont_a10_b100_img = cv2.imread(os.path.join(testsave, 'cont_a10_b100_test.JPG'))
    cont_a10_b100_img = cv2.cvtColor(cont_a10_b100_img, cv2.COLOR_BGR2RGB)
    cont_a15_b0_img = cv2.imread(os.path.join(testsave, 'cont_a15_b0_test.JPG'))
    cont_a15_b0_img = cv2.cvtColor(cont_a15_b0_img, cv2.COLOR_BGR2RGB)
    cont_a15_b25_img = cv2.imread(os.path.join(testsave, 'cont_a15_b25_test.JPG'))
    cont_a15_b25_img = cv2.cvtColor(cont_a15_b25_img, cv2.COLOR_BGR2RGB)
    cont_a15_b50_img = cv2.imread(os.path.join(testsave, 'cont_a15_b50_test.JPG'))
    cont_a15_b50_img = cv2.cvtColor(cont_a15_b50_img, cv2.COLOR_BGR2RGB)
    cont_a15_b75_img = cv2.imread(os.path.join(testsave, 'cont_a15_b75_test.JPG')) 
    cont_a15_b75_img = cv2.cvtColor(cont_a15_b75_img, cv2.COLOR_BGR2RGB)
    cont_a15_b100_img = cv2.imread(os.path.join(testsave, 'cont_a15_b100_test.JPG'))
    cont_a15_b100_img = cv2.cvtColor(cont_a15_b100_img, cv2.COLOR_BGR2RGB)
    cont_a20_b0_img = cv2.imread(os.path.join(testsave, 'cont_a20_b0_test.JPG'))
    cont_a20_b0_img = cv2.cvtColor(cont_a20_b0_img, cv2.COLOR_BGR2RGB)
    cont_a20_b25_img = cv2.imread(os.path.join(testsave, 'cont_a20_b25_test.JPG'))
    cont_a20_b25_img = cv2.cvtColor(cont_a20_b25_img, cv2.COLOR_BGR2RGB)
    cont_a20_b50_img = cv2.imread(os.path.join(testsave, 'cont_a20_b50_test.JPG'))
    cont_a20_b50_img = cv2.cvtColor(cont_a20_b50_img, cv2.COLOR_BGR2RGB)
    cont_a20_b75_img = cv2.imread(os.path.join(testsave, 'cont_a20_b75_test.JPG'))
    cont_a20_b75_img = cv2.cvtColor(cont_a20_b75_img, cv2.COLOR_BGR2RGB)
    cont_a20_b100_img = cv2.imread(os.path.join(testsave, 'cont_a20_b100_test.JPG'))
    cont_a20_b100_img = cv2.cvtColor(cont_a20_b100_img, cv2.COLOR_BGR2RGB)
    cont_a25_b0_img = cv2.imread(os.path.join(testsave, 'cont_a25_b0_test.JPG'))
    cont_a25_b0_img = cv2.cvtColor(cont_a25_b0_img, cv2.COLOR_BGR2RGB)
    cont_a25_b25_img = cv2.imread(os.path.join(testsave, 'cont_a25_b25_test.JPG'))
    cont_a25_b25_img = cv2.cvtColor(cont_a25_b25_img, cv2.COLOR_BGR2RGB)
    cont_a25_b50_img = cv2.imread(os.path.join(testsave, 'cont_a25_b50_test.JPG'))
    cont_a25_b50_img = cv2.cvtColor(cont_a25_b50_img, cv2.COLOR_BGR2RGB)
    cont_a25_b75_img = cv2.imread(os.path.join(testsave, 'cont_a25_b75_test.JPG'))
    cont_a25_b75_img = cv2.cvtColor(cont_a25_b75_img, cv2.COLOR_BGR2RGB)
    cont_a25_b100_img = cv2.imread(os.path.join(testsave, 'cont_a25_b100_test.JPG'))
    cont_a25_b100_img = cv2.cvtColor(cont_a25_b100_img, cv2.COLOR_BGR2RGB)
    cont_a30_b0_img = cv2.imread(os.path.join(testsave, 'cont_a30_b0_test.JPG'))
    cont_a30_b0_img = cv2.cvtColor(cont_a30_b0_img, cv2.COLOR_BGR2RGB)
    cont_a30_b25_img = cv2.imread(os.path.join(testsave, 'cont_a30_b25_test.JPG')) 
    cont_a30_b25_img = cv2.cvtColor(cont_a30_b25_img, cv2.COLOR_BGR2RGB)
    cont_a30_b50_img = cv2.imread(os.path.join(testsave, 'cont_a30_b50_test.JPG'))
    cont_a30_b50_img = cv2.cvtColor(cont_a30_b50_img, cv2.COLOR_BGR2RGB)
    cont_a30_b75_img = cv2.imread(os.path.join(testsave, 'cont_a30_b75_test.JPG')) 
    cont_a30_b75_img = cv2.cvtColor(cont_a30_b75_img, cv2.COLOR_BGR2RGB)
    cont_a30_b100_img = cv2.imread(os.path.join(testsave, 'cont_a30_b100_test.JPG'))
    cont_a30_b100_img = cv2.cvtColor(cont_a30_b100_img, cv2.COLOR_BGR2RGB)
    # cs_fig and cs_ax are used to set up plot for multiple images, with cs_fig used for the figures and cs_ax used to label the images; the images were plotted in too sets because it was large amount for one image
    cs_fig1 = plt.figure()
    cs_ax1 = []
    # plotting the images and their respected titles
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 1))
    cs_ax1[-1].set_title("original")
    plt.imshow(org_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 2))
    cs_ax1[-1].set_title("cont a=1 b=0")
    plt.imshow(cont_a10_b0_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 3))
    cs_ax1[-1].set_title("cont a=1 b=25")
    plt.imshow(cont_a10_b25_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 4))
    cs_ax1[-1].set_title("cont a=1 b=50")
    plt.imshow(cont_a10_b50_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 5))
    cs_ax1[-1].set_title("cont a=1 b=75")
    plt.imshow(cont_a10_b75_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 6))
    cs_ax1[-1].set_title("cont a=1 b=100")
    plt.imshow(cont_a10_b100_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 7))
    cs_ax1[-1].set_title("cont a=1.5 b=0")
    plt.imshow(cont_a15_b0_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 8))
    cs_ax1[-1].set_title("cont a=1.5 b=25")
    plt.imshow(cont_a15_b25_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 9))
    cs_ax1[-1].set_title("cont a=1.5 b=50")
    plt.imshow(cont_a15_b50_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 10))
    cs_ax1[-1].set_title("cont a=1.5 b=75")
    plt.imshow(cont_a15_b75_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 11))
    cs_ax1[-1].set_title("cont a=1.5 b=100")
    plt.imshow(cont_a15_b100_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 12))
    cs_ax1[-1].set_title("cont a=2 b=0")
    plt.imshow(cont_a20_b0_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 13))
    cs_ax1[-1].set_title("cont a=2 b=25")
    plt.imshow(cont_a20_b25_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 14))
    cs_ax1[-1].set_title("cont a=2 b=50")
    plt.imshow(cont_a20_b50_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 15))
    cs_ax1[-1].set_title("cont a=2 b=75")
    plt.imshow(cont_a20_b75_img)
    cs_ax1.append(cs_fig1.add_subplot(4, 4, 16))
    cs_ax1[-1].set_title("cont a=2 b=100")
    plt.imshow(cont_a20_b100_img)
    cs_fig1.tight_layout(pad=1.0, w_pad=1.0, h_pad=2.0)
    plt.show()
    # second set of images for saturation and contrasted images
    cs_fig2 = plt.figure()
    cs_ax2 = []
    cs_ax2.append(cs_fig2.add_subplot(3, 4, 1))
    cs_ax2[-1].set_title("cont a=2.5 b=0")
    plt.imshow(cont_a25_b0_img)
    cs_ax2.append(cs_fig2.add_subplot(3, 4, 2))
    cs_ax2[-1].set_title("cont a=2.5 b=25")
    plt.imshow(cont_a25_b25_img)
    cs_ax2.append(cs_fig2.add_subplot(3, 4, 3))
    cs_ax2[-1].set_title("cont a=2.5 b=50")
    plt.imshow(cont_a25_b50_img)
    cs_ax2.append(cs_fig2.add_subplot(3, 4, 4))
    cs_ax2[-1].set_title("cont a=2.5 b=75")
    plt.imshow(cont_a25_b75_img)
    cs_ax2.append(cs_fig2.add_subplot(3, 4, 5))
    cs_ax2[-1].set_title("cont a=2.5 b=100")
    plt.imshow(cont_a25_b100_img)
    cs_ax2.append(cs_fig2.add_subplot(3, 4, 6))
    cs_ax2[-1].set_title("cont a=3 b=0")
    plt.imshow(cont_a30_b0_img)
    cs_ax2.append(cs_fig2.add_subplot(3, 4, 7))
    cs_ax2[-1].set_title("cont a=3 b=25")
    plt.imshow(cont_a30_b25_img)
    cs_ax2.append(cs_fig2.add_subplot(3, 4, 8))
    cs_ax2[-1].set_title("cont a=3 b=50")
    plt.imshow(cont_a30_b50_img)
    cs_ax2.append(cs_fig2.add_subplot(3, 4, 9))
    cs_ax2[-1].set_title("cont a=3 b=75")
    plt.imshow(cont_a30_b75_img)
    cs_ax2.append(cs_fig2.add_subplot(3, 4, 10))
    cs_ax2[-1].set_title("cont a=3 b=100")
    plt.imshow(cont_a30_b100_img)
    cs_fig2.tight_layout(pad=1.0, w_pad=1.0, h_pad=2.0)
    plt.show()

'''
flipping_image function

This funcation reads in images according to loadind_path, and flips them on vertical plane using fliplr function from numpy. Then save the image according to saving_path.
'''

def flipping_image(loading_path, saving_path):
    data = []
    for filename in (os.listdir(loading_path)):
        path_name = loading_path + filename
        try: 
            img = cv2.imread(path_name)
            if img is not None:
                if ('flip' not in filename):
                    data.append(filename)
        except IOError: 
            pass   
    for i in range(len(data)):
        path = loading_path+data[i]
        img = cv2.imread(path)
        flip = np.fliplr(img)
        img_name = saving_path + 'flip_' + data[i]
        flip = cv2.cvtColor(flip, cv2.COLOR_RGB2BGR)
        mpimg.imsave(img_name, flip)

'''
rotating_image function

This funcation reads in images, and rotates according to given angle, using getRotationMatrix2D and warpAffine functions from opencv4. Function transform.rotate from skimage was originally used, but when opencv and numpy updated, there were conflicts with skimage. Then save the image according to saving_path.
'''

def rotating_image(loading_path, saving_path, given_angle):
    data = []
    
    for filename in (os.listdir(loading_path)):
        path_name = loading_path + filename
        try: 
            img = cv2.imread(path_name)
            if img is not None:
                if ('rot' not in filename):
                    data.append(filename)
        except IOError: 
            pass  

        for i in range(len(data)):
            path = loading_path + data[i]
            img = cv2.imread(path)
            rows = img.shape[0]
            cols = img.shape[1]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), given_angle, 1)
            rot = cv2.warpAffine(img, M, (cols,rows))
            rot_name = saving_path + 'rot'+ str(int(given_angle)) + '_' + data[i]
            rot = cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)
            mpimg.imsave(rot_name, rot)

'''
salt_pepper_image function

This function reads in images, and according to the amount of salt and pepper intenseity, using random coordination, replace some pixels in the image with either a salt or pepper. Then save the image according to saving_path.
'''

def salt_pepper_image(loading_path, saving_path, salt_pepper, amount):
    data = []
    
    for filename in (os.listdir(loading_path)):
        path_name = loading_path + filename
        try: 
            img = cv2.imread(path_name)
            if img is not None:
                if ('sp' not in filename):
                    data.append(filename)
        except IOError: 
            pass  

    for i in range(len(data)):
        path = loading_path + data[i]
        img = cv2.imread(path)
        sp_img = np.copy(img)
        num_salt = np.ceil(amount * img.size * salt_pepper)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        sp_img[coords] = 1
        num_pepper = np.ceil(amount* img.size * (1. - salt_pepper))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        sp_img[coords] = 0
        sp_img = cv2.cvtColor(sp_img, cv2.COLOR_BGR2RGB)
        sp_img_name = saving_path + 'sp_r' + str(int(salt_pepper*100)) + '_a' + str(int(amount*100)) + '_' + data[i]
        sp_img = cv2.cvtColor(sp_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sp_img_name, sp_img)

'''
saturate_contrast_image function

This function reads in images, and according to the alpha(contrast, ranging from 1 to 3) and beta(brightness, ranging from 0 to 100) provided, it will apply it to the image using clip function from numpy. Then save the image according to saving_path. 
'''

def saturate_contrast_image(loading_path, saving_path, alpha, beta):
    data = []
    
    for filename in (os.listdir(loading_path)):
        path_name = loading_path + filename
        try: 
            img = cv2.imread(path_name)
            if img is not None:
                if ('cont' not in filename):
                    data.append(filename)
        except IOError: 
            pass  
        
    for i in range(len(data)):
        path = loading_path + data[i]
        img = cv2.imread(path)
        con_img = np.zeros(img.shape, img.dtype)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                for c in range(img.shape[2]):
                    con_img[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
        con_img_name = saving_path + 'cont_a' + str(int(alpha*10)) + '_b' + str(int(beta)) + '_'  + data[i]
        cv2.imwrite(con_img_name, con_img)

# END OF CODE 


