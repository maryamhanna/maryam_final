#!/usr/bin/env python
# coding: utf-8

'''
DATA AUGMENTATION MAIN CODE

Created by: Maryam Hanna
Date: December 25, 2018
Email: maryamhanna@hotmail.com

This is the main code of data augmentation. It imports files Data_Augmentation_Functions. It is set up, where the user only has to adjust the the loading path, saving path, temperaily path, and whether they want to see test_data for comparision. Further more they have to decide what type of data augmentation they want, which will take multiple types of data for single augmentation. The sections that requires modification is labled below. 
'''

# importing required libraries
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
# importing functions from Data_Augmentation_Functions.py; has to be located in the same directory as this py file
from Data_Augmentation_Functions import *

# if user wishes to view different types of augmentations and their effect, than test_data should be True; otherwise, make it False
test_output = False
if test_output:
    test_data() # function from Data_Augmentation_Functions code 

# user should decide which augmentation they would like to use. If multiple augmentations are true, then the output will also include individual and combined augmentations methods
flip = True # flipping the image about the vertical plane
rot = True # rotating the image certain degrees from the right horizantal plane
salt_pepp = True # adding salt and pepper to image to minimic noise
sat_con = True # saturating or contrasting images to minimic day light conditions

# user should modify the locations of the loading, saving, and temporary path
loadpath = 'C:/Users/wicip/Documents/transfer-master/test_load/'
savepath = 'C:/Users/wicip/Documents/transfer-master/test_save/'
temppath = 'C:/Users/wicip/Documents/transfer-master/test_temp/'
# user can add as many augmentation conditions they wish
rot_ang = [15, 30, 45] # rotation angles in degrees 
amount_sp = [0.08, 0.06] # amount of salt and pepper, range from 0.01 to 0.1
s_n_p = [0.25, 0.5, 0.75] # ratio of salt and pepper, salt/pepper
sc_alpha = [1, 2, 3] # alpha (contrast) of saturation/contrast, range from 1 to 3
sc_beta = [0, 25, 50, 75, 100] # beta (brightness) of saturation/contrast, range from 0 to 100

# concerning the conditions the user inputs, certain functions will occur to load, augment, and save the images properly 
# if the user wants all conditions, flipping, rotating, adding salt and pepper, and saturating and contrasting the images allows the follow code to executed, where functions will be called from Data_Augmentation_Functions is used
if (flip and rot and salt_pepp and sat_con):
    for i in range(len(rot_ang)):
        rotating_image(loadpath, savepath, rot_ang[i])
        rotating_image(loadpath, temppath, rot_ang[i])
    for j in range(len(amount_sp)):
        for k in range(len(s_n_p)):
            salt_pepper_image(loadpath, savepath, s_n_p[k], amount_sp[j])
            salt_pepper_image(temppath, savepath, s_n_p[k], amount_sp[j])
            salt_pepper_image(temppath, temppath, s_n_p[k], amount_sp[j])
            salt_pepper_image(loadpath, temppath, s_n_p[k], amount_sp[j])
    for l in range(len(sc_alpha)):
        for m in range(len(sc_beta)):
            saturate_contrast_image(loadpath, savepath, sc_alpha[l], sc_beta[m])
            saturate_contrast_image(temppath, savepath, sc_alpha[l], sc_beta[m])
    flipping_image(savepath, savepath)
    flipping_image(loadpath, savepath)
# if the user wants only flipped, rotating and adding salt and pepper to the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (flip and rot and salt_pepp and not sat_con):
    for i in range(len(rot_ang)):
        rotating_image(loadpath, savepath, rot[i])
        rotating_image(loadpath, temppath, rot[i])
    for j in range(len(amount_sp)):
        for k in range(len(s_n_p)):
            salt_pepper_image(loadpath, savepath, s_n_p[k], amount_sp[j])
            salt_pepper_image(temppath, savepath, s_n_p[k], amount_sp[j])
    flipping_image(savepath, savepath)
    flipping_image(loadpath, savepath)
# if the user wants only flipped, rotating and saturate and contrast the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (flip and rot and not salt_pepp and sat_con):
    for i in range(len(rot_ang)):
        rotating_image(loadpath, savepath, rot_ang[i])
        rotating_image(loadpath, temppath, rot_ang[i])
    for l in range(len(sc_alpha)):
        for m in range(len(sc_beta)):
            saturate_contrast_image(loadpath, savepath, sc_alpha[l], sc_beta[m])
            saturate_contrast_image(temppath, savepath, sc_alpha[l], sc_beta[m])
    flipping_image(savepath, savepath)
    flipping_image(loadpath, savepath)
# if the user wants only flipped, adding salt and pepper, and saturate and contrast the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (flip and not rot and salt_pepp and sat_con):
    for j in range(len(amount_sp)):
        for k in range(len(s_n_p)):
            salt_pepper_image(loadpath, savepath, s_n_p[k], amount_sp[j])
            salt_pepper_image(loadpath, temppath, s_n_p[k], amount_sp[j])
    for l in range(len(sc_alpha)):
        for m in range(len(sc_beta)):
            saturate_contrast_image(loadpath, savepath, sc_alpha[l], sc_beta[m])
            saturate_contrast_image(temppath, savepath, sc_alpha[l], sc_beta[m])
    flipping_image(savepath, savepath)
    flipping_image(loadpath, savepath)
# if the user wants only rotating, and salt and pepper, and saturate and contrast the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (not flip and rot and salt_pepp and sat_con):
    for i in range(len(rot_ang)):
        rotating_image(loadpath, savepath, rot_ang[i])
        rotating_image(loadpath, temppath, rot_ang[i])
    for j in range(len(amount_sp)):
        for k in range(len(s_n_p)):
            salt_pepper_image(loadpath, savepath, s_n_p[k], amount_sp[j])
            salt_pepper_image(temppath, savepath, s_n_p[k], amount_sp[j])
            salt_pepper_image(temppath, temppath, s_n_p[k], amount_sp[j])
            salt_pepper_image(loadpath, temppath, s_n_p[k], amount_sp[j])
    for l in range(len(sc_alpha)):
        for m in range(len(sc_beta)):
            saturate_contrast_image(loadpath, savepath, sc_alpha[l], sc_beta[m])
            saturate_contrast_image(temppath, savepath, sc_alpha[l], sc_beta[m]) 
# if the user wants only flipped and rotating the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (flip and rot and not salt_pepp and not sat_con):
    for i in range(len(rot_ang)):
        rotating_image(loadpath, savepath, rot_ang[i])
    flipping_image(savepath, savepath)
    flipping_image(loadpath, savepath)
# if the user wants only flipped, and salt and pepper the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (flip and not rot and salt_pepp and not sat_con):
    for j in range(len(amount_sp)):
        for k in range(len(s_n_p)):
            salt_pepper_image(loadpath, savepath, s_n_p[k], amount_sp[j])
    flipping_image(savepath, savepath)
    flipping_image(loadpath, savepath)
# if the user wants only rotating and salt and pepper the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (not flip and rot and salt_pepp and not sat_con):
    for i in range(len(rot_ang)):
        rotating_image(loadpath, savepath, rot_ang[i])
        rotating_image(loadpath, temppath, rot_ang[i])
    for j in range(len(amount_sp)):
        for k in range(len(s_n_p)):
            salt_pepper_image(loadpath, savepath, s_n_p[k], amount_sp[j])
            salt_pepper_image(temppath, savepath, s_n_p[k], amount_sp[j])
# if the user wants only flipped and saturate and contrast the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (flip and not rot and not salt_pepp and sat_con):
    for l in range(len(sc_level)):
        saturate_contrast_image(loadpath, savepath, sc_level[l])
    flipping_image(savepath, savepath)
    flipping_image(loadpath, savepath)
# if the user wants only rotating and saturate and contrast the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (not flip and rot and not salt_pepp and sat_con):
    for i in range(len(rot_ang)):
        rotating_image(loadpath, savepath, rot_ang[i])
        rotating_image(loadpath, temppath, rot_ang[i])
    for l in range(len(sc_alpha)):
        for m in range(len(sc_beta)):
            saturate_contrast_image(loadpath, savepath, sc_alpha[l], sc_beta[m])
            saturate_contrast_image(temppath, savepath, sc_alpha[l], sc_beta[m])
# if the user wants only adding salt and papper, and saturate and contrast the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (not flip and not rot and salt_pepp and sat_con):
    for j in range(len(amount_sp)):
        for k in range(len(s_n_p)):
            salt_pepper_image(loadpath, savepath, s_n_p[k], amount_sp[j])
            salt_pepper_image(loadpath, temppath, s_n_p[k], amount_sp[j])
    for l in range(len(sc_alpha)):
        for m in range(len(sc_beta)):
            saturate_contrast_image(loadpath, savepath, sc_alpha[l], sc_beta[m])
            saturate_contrast_image(temppath, savepath, sc_alpha[l], sc_beta[m])
# if the user wants only flipped the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (flip and not rot and not salt_pepp and not sat_con):
    flipping_image(loadpath, savepath)
# if the user wants only rotating the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (not flip and rot and not salt_pepp and not sat_con):
    for i in range(len(rot_ang)):
        rotating_image(loadpath, savepath, rot_ang[i])  
# if the user wants only add salt and pepper the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (not flip and not rot and salt_pepp and not sat_con):
    for j in range(len(amount_sp)):
        for k in range(len(s_n_p)):
            salt_pepper_image(loadpath, savepath, s_n_p[k], amount_sp[j])
# if the user wants only saturating and contrast the image, the following code will execute, where functions will be called from Data_Augmentation_Functions is used
elif (not flip and not rot and not salt_pepp and sat_con):
    for l in range(len(sc_alpha)):
        for m in range(len(sc_beta)):
            saturate_contrast_image(loadpath, savepath, sc_alpha[l], sc_beta[m])

# END OF CODE 
