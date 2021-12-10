#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 16:10:14 2021

Frame Generation for UCF-101 dataset
Train: first 80% video file
Test: last 20% video file

@author: navin
"""

import os
from cv2 import cv2

FRAME_FOLDER = '/home/navin/UCF-101/'
SAVE_PATH_TRAIN = '/home/navin/paper_5/dataset_ucf/frame/'
SAVE_PATH_TEST = '/home/navin/paper_5/dataset_ucf/frame_test/'
#%%
def seq_to_png(save_path, frame_number, image, folder_name, sub_folder_name):
    '''Read Video file and save each frame as png'''
    if not os.path.exists(save_path+ folder_name):
        os.mkdir(save_path+ folder_name)
    if not os.path.exists(save_path+ folder_name +'/' + sub_folder_name):
        os.mkdir(save_path+ folder_name + '/' + sub_folder_name)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imwrite('{}/{}/{}.png'.format(save_path+ folder_name, sub_folder, frame_number), image)
#%%
UCF_101_FOLDER_LIST = os.listdir(FRAME_FOLDER)
for ucf_each_folder in UCF_101_FOLDER_LIST:
    video_list = os.listdir(FRAME_FOLDER + ucf_each_folder)
    sorted_video_list = sorted(video_list)
    # Training Dataset
    for video_file in sorted_video_list[0: int(0.8 * len(sorted_video_list))]:
        _, _, sub_folder, _ = video_file.split('_')
        cap = cv2.VideoCapture(FRAME_FOLDER + ucf_each_folder + '/' + video_file)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            seq_to_png(SAVE_PATH_TRAIN, count, frame, ucf_each_folder, sub_folder)
            count += 1
    # Testing Dataset
    for video_file in sorted_video_list[int(0.8 * len(sorted_video_list)):]:
        _, _, sub_folder, _ = video_file.split('_')
        cap = cv2.VideoCapture(FRAME_FOLDER + ucf_each_folder + '/' + video_file)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            seq_to_png(SAVE_PATH_TEST, count, frame, ucf_each_folder, sub_folder)
            count += 1
            