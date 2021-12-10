#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 16:10:14 2021

@author: navin
"""

import os
#import glob
from cv2 import cv2
#import matplotlib.pyplot as plt
FRAME_FOLDER = '/home/navin/data1/'
SAVE_PATH = '/home/navin/paper_5/DATASET_10/frame/'
#SAVE_PATH_OPTICAL = '/home/navin/paper_5/dataset1/png_optical_all/'
#%%
frame_folder_list = os.listdir(FRAME_FOLDER)
frame_folder_list.sort()
#%%
def seq_to_png(frame_number, image):
    '''Read Video file and save each frame as png'''
#    if not os.path.exists(SAVE_PATH):
#        os.mkdir(SAVE_PATH + set_number)
    image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
    cv2.imwrite('{}/{}.png'.format(SAVE_PATH, frame_number), image)
#    pass
count = 0
for folder in frame_folder_list:
    seq_list = os.listdir(FRAME_FOLDER + folder)
    seq_list.sort()
    for seq in seq_list:
#        for set_name in sorted(glob.glob('{}/*.seq'.format(data_path))):
        print(FRAME_FOLDER + folder + '/' + seq)
        cap = cv2.VideoCapture(FRAME_FOLDER + folder + '/' + seq)
#        print(cap)
        while True:
#            count += 1
            ret, frame = cap.read()
            if not ret:
                break
#            if count == 5:
            seq_to_png(count, frame )
            count += 1
#%%
#'''optical flow generator'''
#DATA_PATH = os.path.join(SAVE_PATH + '/' + 'set00')
#DATA_LIST = os.listdir(DATA_PATH)
#DATA_LIST = sorted(DATA_LIST, key=lambda x:int(x.split('.')[0]))
#for each in range (len(DATA_LIST)):
#    frame1 = cv2.imread(os.path.join(DATA_PATH , str(each) + '.png'), 0)
#    frame2 = cv2.imread(os.path.join(DATA_PATH , str(each) + '.png'), 0)
#    optical_flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 1, 15, 3, 1.1, 0)
#    cv2.imwrite(SAVE_PATH_OPTICAL + '/' + '{}.png'.format(str(each)), optical_flow[:,:,0]*255)
#%%
    

    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
