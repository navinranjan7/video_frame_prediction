#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:06:38 2021

@author: navin
"""
import os
from cv2 import cv2

SAVE_PATH_TRAIN = '/home/navin/paper_5/dataset_ucf/optical_flow/train/'
SAVE_PATH_TEST = '/home/navin/paper_5/dataset_ucf/optical_flow/test/'
DATA_PATH_TRAIN = '/home/navin/paper_5/dataset_ucf/frame/'
DATA_PATH_TEST = '/home/navin/paper_5/dataset_ucf/frame_test/'
#%%
def save_to_png(save_path, frame_number, image, folder_name, sub_folder_name):
    '''Read Video file and save each frame as png'''
    if not os.path.exists(save_path + folder_name):
        os.mkdir(save_path + folder_name)
    if not os.path.exists(save_path + folder_name +'/' + sub_folder_name):
        os.mkdir(save_path + folder_name + '/' + sub_folder_name)
    cv2.imwrite('{}/{}/{}.png'.format(
        save_path + folder_name, sub_folder_name, frame_number), image)
    #%%
def frame_to_opticalflow(folder_path, save_path):
    '''
    function to generate optical flow using two frames
    '''
    frame_category_list = os.listdir(DATA_PATH_TRAIN)
    for category in frame_category_list:
        data_group = os.listdir(os.path.join(folder_path, category))
        for each_group in sorted(data_group):
            frame_list = os.listdir(os.path.join(folder_path, category, each_group))
            for frame_number in range(0, len(frame_list) -1):
                frame_1 = cv2.imread(os.path.join(
                    folder_path, category, each_group, str(frame_number) + '.png'))
                frame_2 = cv2.imread(os.path.join(
                    folder_path, category, each_group, str(frame_number + 1) + '.png'))
                frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
                frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
                optical_flow = cv2.calcOpticalFlowFarneback(
                    frame_1, frame_2, None, 0.5, 3, 15, 3, 5, 1.1, 0)
                magnitude, _ = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
                save_to_png(save_path, frame_number, magnitude, category, each_group)
#%%
frame_to_opticalflow(DATA_PATH_TRAIN, SAVE_PATH_TRAIN)
frame_to_opticalflow(DATA_PATH_TEST, SAVE_PATH_TEST)
#%%
               