#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 16:01:29 2021

@author: navin
"""

import tensorflow as tf
import numpy as np
import cv2
def gradient_diff_loss(y_true, y_pred):
    total_gradient = 0
    padding_row = tf.constant([[0,0],[0,1]])
    padding_col = tf.constant([[1,0],[0,0]])
    for sample in range(y_true.shape[0]):
        for channel in range(y_true.shape[3]):
            true = y_true[sample, :, :, channel]
            pred = y_pred[sample, :, :, channel]
            
            true_shift_col = true[1:]
            true_shift_col_pad = tf.pad(true_shift_col, padding_col, "CONSTANT")
            true_shift_row = true[:,1:]
            true_shift_row_pad = tf.pad(true_shift_row, padding_row, "CONSTANT")
            
            pred_shift_col = pred[1:]
            pred_shift_col_pad = tf.pad(pred_shift_col, padding_col, "CONSTANT")
            pred_shift_row = pred[:,1:]
            pred_shift_row_pad = tf.pad(pred_shift_row, padding_row, "CONSTANT")
            
            true_gradient_row = tf.math.abs(tf.math.subtract(true, true_shift_row_pad))
            pred_gradient_row = tf.math.abs(tf.math.subtract(pred, pred_shift_row_pad))
            
            gradient_row = tf.math.abs(tf.math.subtract(true_gradient_row, pred_gradient_row))
            
            true_gradient_col = tf.math.abs(tf.math.subtract(true, true_shift_col_pad))
            pred_gradient_col = tf.math.abs(tf.math.subtract(pred, pred_shift_col_pad))
            
            gradient_col = tf.math.abs(tf.math.subtract(true_gradient_col, pred_gradient_col))
            
            gradient_row_col = tf.math.abs(tf.math.add(gradient_row, gradient_col))
            gradient_row_col = tf.reduce_sum(gradient_row_col,[0,1])
            total_gradient += gradient_row_col
#        total_gradient_normalized = total_gradient/(4*640*480)
    total_gradient_normalized = total_gradient/(y_true.shape[1] * y_true.shape[2] * y_true.shape[3])
    return (total_gradient_normalized) 
#%%       
def squared_distance_loss(y_true, y_pred):
#    mean_square_error = tf.keras.metrics.mean_squared_error(y_true, y_pred)
#    mean_square_error = tf.reduce_sum(mean_square_error)
    total_squared_error = 0
    for sample in range(y_true.shape[0]):
        for channel in range(y_true.shape[3]):
            true = y_true[sample, :, :, channel]
            pred = y_pred[sample, :, :, channel]
            error = tf.math.subtract(true, pred)
            error_squared = tf.math.square(error)
            error_squared_sum = tf.reduce_sum(error_squared,[0,1])
            total_squared_error += error_squared_sum
    squared_error_normalized = total_squared_error/(y_true.shape[1] * y_true.shape[2] * y_true.shape[3])    
    return squared_error_normalized
#%%%
def absolute_distance_loss(y_true, y_pred):
    total_abs_error = 0
    for sample in range(y_true.shape[0]):
        for channel in range(y_true.shape[3]):
            true = y_true[sample, :, :, channel]
            pred = y_pred[sample, :, :, channel]
            error = tf.math.subtract(true, pred)
            error_abs = tf.math.abs(error)
            error_abs_sum = tf.reduce_sum(error_abs,[0,1])
            total_abs_error += error_abs_sum
    abs_error_normalized = total_abs_error/( y_true.shape[1] * y_true.shape[2] * y_true.shape[3])    
    return abs_error_normalized
#%%
def ssim_loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val = 1, filter_size = 11,
                         filter_sigma=1.5, k1=0.01, k2=0.03)
    ssim = tf.reduce_sum(ssim)
    ssim = 1 - (ssim/(y_true.shape[0]))
    return ssim * (y_true.shape[0])

#%%
def optical_flow_loss(y_prev, y_true, y_pred):
    y_prev = y_prev.numpy()
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    optical_loss = 0
    for sample in range(0, y_true.shape[0]):
        optical_flow_true = np.zeros(shape = (256, 256, 2))
        optical_flow_pred = np.zeros(shape = (256, 256, 2))
        frame_prev = y_prev[sample,:,:,:]
        frame_true = y_true[sample,:,:,:]
        frame_pred = y_pred[sample,:,:,:]
#        print(optical_flow_ori_mag_ang.shape)
        for channel in range (0,frame_prev.shape[2]):
            optical_flow_true += cv2.calcOpticalFlowFarneback(frame_prev[...,channel], frame_true[...,channel], None, 0.5, 3, 15, 3, 5, 1.1, 0)   
            optical_flow_pred += cv2.calcOpticalFlowFarneback(frame_prev[...,channel], frame_pred[...,channel], None, 0.5, 3, 15, 3, 5, 1.1, 0)   
        optical_flow_true = cv2.normalize(optical_flow_true, None, 0, 1, cv2.NORM_MINMAX)
        magnitude_true, _ = cv2.cartToPolar(optical_flow_true[...,0], optical_flow_true[...,1])
        optical_flow_pred = cv2.normalize(optical_flow_pred, None, 0, 1, cv2.NORM_MINMAX)
        magnitude_pred, _ = cv2.cartToPolar(optical_flow_pred[...,0], optical_flow_pred[...,1])
        mae = tf.keras.metrics.mean_absolute_error(magnitude_true, magnitude_pred)
        mae = tf.reduce_sum(mae)
        optical_loss += mae
    optical_loss = optical_loss/y_true.shape[0]
    return optical_loss
#%%
def losses(y_prev, y_true, y_pred, gdl=False, sdl=False, adl=False, ssim=False, optical_flow=False):
#    y_true = (y_true*127.5)+127.5
#    y_pred = (y_pred*127.5)+127.5
    gdl_loss, sdl_loss, mae_loss, ssim_loss1, optical_loss = 0, 0, 0, 0, 0
    if gdl:
        gdl_loss = gradient_diff_loss(y_true, y_pred)
    if sdl:
        sdl_loss = squared_distance_loss(y_true, y_pred)
    if ssim:
        ssim_loss1 = ssim_loss(y_true, y_pred)
    if adl:
        mae_loss = absolute_distance_loss(y_true, y_pred)
    if optical_flow:
        optical_loss = optical_flow_loss(y_prev, y_true, y_pred)
    return (gdl_loss, sdl_loss, mae_loss, ssim_loss1, optical_loss)
#%%
#data1 = cv2.imread('/home/navin/paper_5/dataset1/png_frame_all/set09/295.png')
#data2 = cv2.imread('/home/navin/paper_5/dataset1/png_frame_all/set09/296.png')
#ori = cv2.imread('/home/navin/paper_5/dataset1/png_optical_all/png_optical_nocart_mag_angle/295.png')
#a,b,c = optical_flow_loss(data1, data2, ori)
##%%
#import matplotlib.pyplot as plt
#plt.imshow(ori[...,0])
##%%
#from dataloader_256_256_stack_1 import MultiInputGenerator
##'''Data Tester'''
#frame_path = '/home/navin/paper_5/dataset1/png_frame_all/set09/'
#optical_path = '/home/navin/paper_5/dataset1/png_optical_all/png_optical_nocart_mag_angle/'
#past_sequence = 2
#prediction_step = 1
#total_sample = 20000
#batch_size = 2
#start = 0
#data_loader = MultiInputGenerator(frame_path, optical_path)
#x_frame, x_optical, output, output_optical = data_loader.get_data(start, past_sequence + 1, prediction_step, batch_size)
##%%
#mae = optical_flow_loss(x_frame[:,:,:,3:6], x_frame[:,:,:,3:6], x_frame[:,:,:,3:6])
#mae
##%%
#plt.imshow(x_frame[1,:,:,3]*255)
#a*255
#x_frame[:,:,:,3:6].shape
