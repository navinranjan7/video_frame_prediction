#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:53:16 2021

@author: navin
"""


import numpy as np 
import cv2
import os

from tensorflow.keras.utils import Sequence
import tensorflow as tf

TARGET_WIDTH, TARGET_HEIGHT = 256, 256


class MultiInputGenerator(Sequence):
    def __init__(self, frame_path, optical_path):
        self.frame_path = frame_path
        self.optical_path = optical_path 
    
    def get_image_batch(self, start_number, end_number, source):
        image_holder = []
#        optical_holder = []
        for each_image in range(start_number, end_number):
            if source == 'frame':
                image = cv2.imread(os.path.join(self.frame_path, f"{each_image}.png"))
#                print(self.frame_path, f"{each_image}.png")
#                image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
                image = image/255.
            elif source == 'optical':
                image = cv2.imread(os.path.join(self.optical_path, f"{each_image}.png"))
#                print(self.optical_path, f"{each_image}.png")
#                image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
                image = image/255.
            image_holder.append(image)
        image_holder = np.concatenate(image_holder, axis=-1)
#        image_holder = image_holder.reshape( 1, image_holder.shape[0], image_holder.shape[1], image_holder.shape[2])
        image_holder = np.array(image_holder)
        image_holder = image_holder.astype('float32')
        

        return image_holder
        
    def get_data(self, start, past_sequence, prediction_step, batch_size):
        frame, optical, output, output_optical = [], [], [], []
        for batch in range(0, batch_size):
#            print(1)
            frame_input = self.get_image_batch(batch + start + 1, 
                                               batch + start + past_sequence,'frame')
#            print(2)
            optical_input = self.get_image_batch(batch + start , 
                                                 batch + start + past_sequence -1,'optical')
#            print(3)
            frame_output =  self.get_image_batch(batch + start + past_sequence + prediction_step -1, 
                                                 batch + start + past_sequence + prediction_step, 'frame')
#            print(4)
            optical_output = self.get_image_batch(batch + start + past_sequence + prediction_step -2, 
                                                  batch + start + past_sequence + prediction_step-1, 'optical')
            frame.append(frame_input)
            optical.append(optical_input)
            output.append(frame_output)
            output_optical.append(optical_output)

        frame = tf.convert_to_tensor(np.array(frame))
        optical = tf.convert_to_tensor(np.array(optical))
        output = tf.convert_to_tensor(np.array(output))
        output_optical = tf.convert_to_tensor(np.array(output_optical))
        return (frame, optical, output, output_optical)
               
#%%
#'''Data Tester'''
#frame_path = '/home/navin/paper_5/dataset1/png_frame_all/set09/'
#optical_path = '/home/navin/paper_5/dataset1/png_optical_all/png_optical_nocart_mag_angle1/'
#past_sequence = 2
#prediction_step = 1
#total_sample = 20000
#batch_size = 2
#start = 0
#data_loader = MultiInputGenerator(frame_path, optical_path)
#x_frame, x_optical, output, output_optical = data_loader.get_data(start, past_sequence + 1, prediction_step, batch_size)
#%%
