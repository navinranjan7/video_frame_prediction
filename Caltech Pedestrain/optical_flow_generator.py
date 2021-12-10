#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:06:38 2021

@author: navin
"""
import matplotlib.pyplot as plt
from cv2 import cv2 
import os
#%%

save_path = '/home/navin/paper_5/DATASET_10/optical_flow/'
data_path = '/home/navin/paper_5/DATASET_10/frame/'


data_list = os.listdir(data_path)
data_list = sorted(data_list, key=lambda x : int(x.split('.')[0]))
data_list
for frame_number in range (0, len(data_list) - 1):
    data1 = cv2.imread(os.path.join(data_path, str(frame_number) + '.png'))
    data2 = cv2.imread(os.path.join(data_path, str(frame_number + 1) + '.png'))
    frame1 = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(data2, cv2.COLOR_BGR2GRAY)
    optical_flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    magnitude, angle = cv2.cartToPolar(optical_flow[...,0], optical_flow[...,1])
    cv2.imwrite(os.path.join(save_path, f"{frame_number}.png"), magnitude)
#%%
data1 = cv2.imread('/home/navin/paper_5/dataset/png/set00/V000/1599.png')
data2 = cv2.imread('/home/navin/paper_5/dataset/png/set00/V000/1600.png')
frame1 = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
frame2 = cv2.cvtColor(data2, cv2.COLOR_BGR2GRAY)
optical_flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.1, 0)   
magnitude = cv2.normalize(optical_flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
#%%
#plt.imshow(magnitude, cmap='jet')
#plt.colorbar()
#plt.show()
#%%

