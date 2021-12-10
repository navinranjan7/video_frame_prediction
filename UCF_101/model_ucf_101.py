#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:36:27 2021

@author: navin
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import numpy as np
import tensorflow as tf

from cv2 import cv2
from tensorflow import keras
from tensorflow.keras import layers

from data_loader_ucf_101 import MultiInputGenerator
from loss import ssim_loss, squared_distance_loss, psnr_loss
from loss1 import losses


FRAME_TRAIN_DIR = '/home/navin/paper_5/dataset_ucf/frame/'
OPTICAL_FLOW_TRAIN_DIR = '/home/navin/paper_5/dataset_ucf/optical_flow/train/'

FRAME_TEST_DIR = '/home/navin/paper_5/dataset_ucf/frame_test/'
OPTICAL_FLOW_TEST_DIR = '/home/navin/paper_5/dataset_ucf/optical_flow/test/'

SAVE_PATH_PREDICTED = '/home/navin/paper_5/dataset_ucf/result/predicted/'
SAVE_PATH_GROUND_TRUTH = '/home/navin/paper_5/dataset_ucf/result/ground_truth/'

STRIDES_CONV = (1, 1)
STRIDES_UP = (2, 2)

ACTIVATION = tf.nn.relu 
PADDING = 'same'

CONV_POOL_SIZE = (2, 2)
CONV_POOL_STRIDES = (2, 2)
KERNEL_SIZE_CONV = (3, 3)
KERNEL_SIZE_RNN = (3, 3)
KERNEL_SIZE_UP = (2, 2)
#%%
class Pooling(layers.Layer):
    ''' Pooling operation on Tensor'''
    def __init__(self, out_channel):
        super(Pooling, self).__init__()
        self.conv_pooling = layers.Conv2D(out_channel,
                                          kernel_size=CONV_POOL_SIZE,
                                          strides=CONV_POOL_STRIDES,
                                          padding='valid',
                                          kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1),
                                          activation=ACTIVATION)
        self.b_n = layers.BatchNormalization()
    def call(self, input_tensor, training=False):
        max_pool = self.conv_pooling(input_tensor, training=training)
#        print(f"Pooling: ", max_pool.shape)
        return max_pool
#%%
class CNNBlock(layers.Layer):
    ''' Generate a CNN block with one convolution, batch normalization and
    relu activation function '''
    def __init__(self, out_channels):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels,
                                  kernel_size=KERNEL_SIZE_CONV,
                                  strides=STRIDES_CONV,
                                  padding=PADDING,
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1),
                                  activation=ACTIVATION)
        self.b_n = layers.BatchNormalization()
        self.dropout = layers.Dropout(rate=0.1)
    def call(self, input_tensor, training=False):
        convolution = self.conv(input_tensor, training=training)
#        print(f"layer :", convolution.shape)
        batch_normalization = self.b_n(convolution, training=training)
        dropout = self.dropout(batch_normalization)
        return dropout
#%%
class EncoderBlock(layers.Layer):
    ''' Two CNN block and one Concat '''
    def __init__(self, channels):
        super(EncoderBlock, self).__init__()
        self.cnn1 = CNNBlock(channels[0])
        self.cnn2 = CNNBlock(channels[1])
        self.concat = layers.Concatenate(axis=-1)

    def call(self, input_tensor, training=False):
        convolution_block1 = self.cnn1(input_tensor, training=training)
        convolution_block2 = self.cnn2(convolution_block1, training=training)
#        convolution_block2 = self.concat([input_tensor, convolution_block2]) 
        return convolution_block2
#%%
class DecoderBlock(layers.Layer):
    '''Upsampling, Concatenate with frame and optical encoder and 2 CNNBlock layer'''
    def __init__(self, channels):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = layers.Conv2DTranspose(channels[0],
                                                     kernel_size=KERNEL_SIZE_UP,
                                                     strides=STRIDES_UP,
                                                     padding='valid',
                                                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1),
                                                     activation=ACTIVATION)
        self.b_n = layers.BatchNormalization()
        self.multiplication = layers.Multiply()
        self.concat = layers.Concatenate(axis=-1)
        self.add = layers.Add()
        self.activation1 = layers.Activation(activation=ACTIVATION)
        self.activation2 = layers.Activation(activation=ACTIVATION)
        self.cnn_block1 = CNNBlock(channels[1])
        self.cnn_block2 = CNNBlock(channels[2])
        self.mask = layers.Conv2D(channels[1],
                          kernel_size=KERNEL_SIZE_CONV,
                          strides=STRIDES_CONV,
                          padding=PADDING,
                          kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1),
                          activation='relu')
        
    def call(self, input_tensor, frame_conn, optical_conn, model_type, optical_feed=False, concatenation=False, training=False):
        up_sampling = self.conv_transpose(input_tensor, training=training)
        up_sampling = self.b_n(up_sampling, training=training)
#        print(f"Up_sampling: ", up_sampling.shape)
        if model_type == 'main':
            concat_frame_optical = self.concat([up_sampling, frame_conn])
            convolution_block = self.cnn_block1(concat_frame_optical, training=training)
            if optical_feed:
                mask = self.mask(convolution_block, training=training)
                mask_mul = self.multiplication([mask, optical_conn])
                add = self.add([convolution_block, mask_mul])
                convolution_block = self.cnn_block2(add, training=training)
        elif model_type == 'optical':
            if concatenation:
                up_sampling = self.concat([up_sampling, optical_conn])
            convolution_block = self.cnn_block1(up_sampling, training=training)
#                print(f"Concat: ", concat_frame_optical.shape)
            convolution_block = self.cnn_block2(convolution_block, training=training)
#                print(f"Concat: ", concat_frame_optical.shape)
        return convolution_block

#%%
class RNNConvLSTM(layers.Layer):
    ''' Two layer of ConvLSTM'''
    def __init__(self, out_channels):
        super(RNNConvLSTM, self).__init__()
        self.rnn_1 = layers.ConvLSTM2D(out_channels[0],
                                       kernel_size=KERNEL_SIZE_RNN,
                                       activation=ACTIVATION,
                                       padding='same',
                                       return_sequences=True)
        self.rnn_2 = layers.ConvLSTM2D(out_channels[1],
                                       kernel_size=KERNEL_SIZE_RNN,
                                       activation=ACTIVATION,
                                       padding='same',
                                       return_sequences=False)
    def call(self, input_tensor, training=False):
        rnn_out = self.rnn_1(input_tensor, training=training)
#        print(f"RNN :", rnn_out.shape)
        rnn_out = self.rnn_2(rnn_out, training=training)
#        print(f"RNN :", rnn_out.shape)
        return rnn_out
#%%
class OpticalModel(keras.Model):
    '''Optical flow Recurrent Convolutional Autoencoder for predicting optical flow '''
    def __init__(self):
        super(OpticalModel, self).__init__()
        #Optical Encoder
        self.optical_block1 = EncoderBlock([32, 32])
        self.optical_block2 = EncoderBlock([64, 64])
        self.optical_block3 = EncoderBlock([128, 128])
        self.optical_block4 = EncoderBlock([256, 256])
        self.optical_block5 = EncoderBlock([512, 512])
        
        self.optical_pool1 = Pooling(64)
        self.optical_pool2 = Pooling(128)
        self.optical_pool3 = Pooling(256)
        self.optical_pool4 = Pooling(512)
        
        self.reshape_rnn_in = layers.Reshape((1, 16, 16, 512))
        self.reshape_rnn_out = layers.Reshape((16, 16, 512))
        
        self.optical_rnn_block1 = RNNConvLSTM([512, 512])
        
        self.decode_block4 = DecoderBlock([256, 256, 256])
        self.decode_block3 = DecoderBlock([128, 128, 128])
        self.decode_block2 = DecoderBlock([64, 64, 64])
        self.decode_block1 = DecoderBlock([32, 32, 32])
        self.out_layer = layers.Conv2D(3,
                                  kernel_size=KERNEL_SIZE_CONV,
                                  strides=STRIDES_CONV,
                                  padding=PADDING,
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1),
                                  activation='sigmoid')
    def call(self, optical_input, training=False):
        e_optical1 = self.optical_block1(optical_input, training=training)
        o_pool1 = self.optical_pool1(e_optical1, training=training)
        ### optical Encoder Block 2
        e_optical2 = self.optical_block2(o_pool1, training=training)
        o_pool2 = self.optical_pool2(e_optical2, training=training)
        ### optical Encoder Block 3
        e_optical3 = self.optical_block3(o_pool2, training=training)
        o_pool3 = self.optical_pool3(e_optical3, training=training)
#        ### optical Encoder Block 4
        e_optical4 = self.optical_block4(o_pool3, training=training)
        o_pool4 = self.optical_pool4(e_optical4, training=training)
        ##
        e_optical5 = self.optical_block5(o_pool4, training=training)
        ### Reshape to fit ConvLSTM2D layer
        reshape_in = self.reshape_rnn_in(e_optical5)
        ### Recurrent Layer
        optical_rnn1 = self.optical_rnn_block1(reshape_in, training=training)
        ###Reshape to fit decoder block 
        optical_rnn1 = self.reshape_rnn_out(optical_rnn1)
        ### Optical Decoder Block 4
        d_optical4 = self.decode_block4(optical_rnn1, e_optical4, e_optical4, 'optical', optical_feed=False, concatenation=True, training=training)
#        ### Optical Decoder Block 3
        d_optical3 = self.decode_block3(d_optical4, e_optical3, e_optical3, 'optical', optical_feed=False, concatenation=True, training=training)
        ### Optical Decoder Block 2
        d_optical2 = self.decode_block2(d_optical3, e_optical2, e_optical2, 'optical', optical_feed=False, concatenation=True, training=training)
        ### Optical Decoder Block 1
        d_optical1 = self.decode_block1(d_optical2, e_optical1, e_optical1, 'optical', optical_feed=False, concatenation=True, training=training)
        
        # output layer
        output = self.out_layer(d_optical1)
        return([output, d_optical4, d_optical3, d_optical2, d_optical1])
#%%        
class ProposedModel(keras.Model):
    '''Proposed Model: 4 blocks of frame encoder, e_optical flow, and decoder
    and two block of convLSTM for frame and optical flow with skip connection '''
    def __init__(self):
        super(ProposedModel, self).__init__()
        # Frame Encoder
        self.frame_block1 = EncoderBlock([32, 32])
        self.frame_block2 = EncoderBlock([64, 64])
        self.frame_block3 = EncoderBlock([128, 128])
        self.frame_block4 = EncoderBlock([256, 256])
        self.frame_block5 = EncoderBlock([512, 512])
        
        self.frame_pool1 = Pooling(64)
        self.frame_pool2 = Pooling(128)
        self.frame_pool3 = Pooling(256)
        self.frame_pool4 = Pooling(512)

        # Optical Model
        self.optical_model = OpticalModel()

        # RNN Model
        self.reshape_rnn_in = layers.Reshape((1, 16, 16, 512))
        self.reshape_rnn_out = layers.Reshape((16, 16, 512))
        self.frame_rnn_block1 = RNNConvLSTM([512, 512])
        self.frame_optical_add = layers.Add()
        
        self.decode_block4 = DecoderBlock([256, 256, 256])
        self.decode_block3 = DecoderBlock([128, 128, 128])
        self.decode_block2 = DecoderBlock([64, 64, 64])
        self.decode_block1 = DecoderBlock([32, 32, 32])
        self.out_layer = layers.Conv2D(3,
                                  kernel_size=KERNEL_SIZE_CONV,
                                  strides=STRIDES_CONV,
                                  padding=PADDING,
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1),
                                  activation='sigmoid')
        self.frame_optical_add1 = layers.Add()
    
    
    def call(self, inputs, training=False, training1=False):
        ''' Proposed model'''
        frame1 = self.frame_block1(inputs[0], training=training)
        f_pool1 = self.frame_pool1(frame1)
        # Frame Encoder Block 2  --> 320x240
        frame2 = self.frame_block2(f_pool1, training=training)
        f_pool2 = self.frame_pool2(frame2)
        # Frame Encoder Block 3  -->
        frame3 = self.frame_block3(f_pool2, training=training)
        f_pool3 = self.frame_pool3(frame3)
        # Frame Encoder Block 4
        frame4 = self.frame_block4(f_pool3, training=training)
        f_pool4 = self.frame_pool4(frame4)
        ##
        frame5 = self.frame_block5(f_pool4, training=training)
        
        _, d_optical4, d_optical3, d_optical2, d_optical1= self.optical_model(inputs[1], training=training1)
        #Reshape to fit to ConvLSTM
        reshape_in = self.reshape_rnn_in(frame5)
        #ConvLSTM to learn timeseries
        frame_rnn1 = self.frame_rnn_block1(reshape_in, training=training)
        ###Reshape to fit decoder block 
        frame_rnn1 = self.reshape_rnn_out(frame_rnn1)
        
        # Model Decoder --> concat frame and optical
        decoder4 = self.decode_block4(frame_rnn1, frame4, d_optical4, 'main',  optical_feed=True, concatenation=True, training=training)
        decoder3 = self.decode_block3(decoder4, frame3, d_optical3, 'main',  optical_feed=True, concatenation=True, training=training)
        decoder2 = self.decode_block2(decoder3, frame2, d_optical2, 'main',  optical_feed=True, concatenation=True, training=training)
        decoder1 = self.decode_block1(decoder2, frame1, d_optical1, 'main',  optical_feed=True, concatenation=True, training=training)

        # output layer
        output = self.out_layer(decoder1)
        return output
#%%   
def compute_loss_main(output_optical, y_true, y_pred):
    loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow = losses(output_optical, y_true, y_pred, 
                                                                                              gdl=True, sdl=True, 
                                                                                              adl=False, ssim=True, 
                                                                                              optical_flow=False)
    loss =  loss_value_gdl + loss_value_lse + loss_value_ssim + loss_value_flow + loss_value_mae
    return loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow, loss

def compute_loss_optical(output_optical, y_true, y_pred):
    loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow = losses(output_optical, y_true, y_pred, 
                                                                                              gdl=False, sdl=False, 
                                                                                              adl=True, ssim=False, 
                                                                                              optical_flow=False)
    loss =  loss_value_gdl + loss_value_lse + loss_value_ssim + loss_value_flow + loss_value_mae
    return loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow, loss
#%%
model = ProposedModel()
model_optical = OpticalModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
optimizer_optical = tf.keras.optimizers.Adam(learning_rate=1e-4)
#%%
@tf.function
def train_main(x_frame, x_optical, output, output_optical):
    with tf.GradientTape() as tape:
        logits = model([x_frame, x_optical], training=True, training1=False)
        loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow, loss = compute_loss_main(output_optical, output, logits)
    grads = tape.gradient(loss , model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow, loss
#%%
@tf.function
def train_optical(x_optical, output_optical):
    with tf.GradientTape() as tape1:
        logits1,_,_,_,_ = model_optical(x_optical, training=True)
        _, _, _, _, _, loss = compute_loss_optical(x_optical, output_optical, logits1)
    grads1 = tape1.gradient(loss , model_optical.trainable_variables)
    optimizer_optical.apply_gradients(zip(grads1, model_optical.trainable_variables))
    return loss
#%%    
def save_as_png(save_path, folder_name, sub_folder_name, frame_number, image):
    '''Read Video file and save each frame as png'''
    if not os.path.exists(save_path+ folder_name):
        os.mkdir(save_path+ folder_name)
    if not os.path.exists(save_path+ folder_name +'/' + sub_folder_name):
        os.mkdir(save_path+ folder_name + '/' + sub_folder_name)
    cv2.imwrite('{}/{}/{}.png'.format(save_path+ folder_name, sub_folder_name, frame_number), image)
#%%
def train_epoch(epochs = 100, batch_size = 8):
    '''
    Train the model based on Training Dataset
    Train the OPtical Flow Model for first 20 epochs
    '''
    past_sequence = 2
    prediction_step = 1
    count_category = 0
    for epoch in range (0, epochs):
        frame_train_folders_list = os.listdir(FRAME_TRAIN_DIR)
        for frame_train_folder in frame_train_folders_list:
            count_category += 1
            data_group = os.listdir(os.path.join(FRAME_TRAIN_DIR, frame_train_folder))
            for each_group in sorted(data_group):
                frame_path = os.path.join(FRAME_TRAIN_DIR, frame_train_folder, each_group)
                optical_path = os.path.join(OPTICAL_FLOW_TRAIN_DIR, frame_train_folder, each_group)
                data_loader = MultiInputGenerator(frame_path, optical_path)
                total_sample = len(os.listdir(frame_path))
                for start in range(0, total_sample - past_sequence - batch_size, batch_size): 
#                    count_category += 1
                    x_frame, x_optical, output, output_optical = data_loader.get_data(start, past_sequence + 1, prediction_step, batch_size)
                    if epoch <= 20:
                        loss_value_opt = train_optical(x_optical, output_optical)
#                    print(f"optical_flow_loss : {loss_value_opt}")
                        loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow, loss_value = train_main(x_frame, x_optical, output, output_optical)
                        if start % 100 == 0:
                            print(f"Epochs {epoch}  {count_category} Training Loss: {float(loss_value)} {loss_value_opt}")
                            print(loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow)
                    else:
                        loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow, loss_value = train_main(x_frame, x_optical, output, output_optical)
                        if start % 100 == 0:
                            print(f"Epochs {epoch}  {count_category} Training Loss: {float(loss_value)}")
                            print(loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow)

#    print(count_category)                    
    model.save_weights(f"/home/navin/paper_5/dataset/Output/model_ucf_101.h5")   
    model_optical.save_weights(f"/home/navin/paper_5/dataset/Output/model_ucf_101_optical_1.h5") 
#%%
def test(batch_size = 8, save_predicted_result=False):
    '''
    Test the trained model on Testing Dataset
    Check the performance meterics, and 
    if save_prediction_result is True
        save the predicted next frame
    '''
    past_sequence = 2
    prediction_step = 1
    ssim_loss_holder = []
    mse_loss_holder = []
    psnr_loss_holder = []    
    frame_test_folders_list = os.listdir(FRAME_TEST_DIR)
    count_category = 0
    for frame_test_folder in frame_test_folders_list:
        data_group = os.listdir(os.path.join(FRAME_TEST_DIR, frame_test_folder))
        for each_group in sorted(data_group):
            frame_path = os.path.join(FRAME_TEST_DIR, frame_test_folder, each_group)
            optical_path = os.path.join(OPTICAL_FLOW_TEST_DIR, frame_test_folder, each_group)
            data_loader_test = MultiInputGenerator(frame_path, optical_path)
            total_sample = len(os.listdir(frame_path))
            for start in range(0, total_sample - past_sequence - batch_size, batch_size): 
                count_category += 1
                x_frame_test, x_optical_test, output_test, _ = data_loader_test.get_data(start, past_sequence + 1, prediction_step, batch_size)
                val_logits = model([x_frame_test, x_optical_test], training=False)
                ssim_loss_holder.append(ssim_loss(output_test, val_logits))
                mse_loss_holder.append(squared_distance_loss(output_test, val_logits))
                psnr_loss_holder.append(psnr_loss(output_test, val_logits))
                if save_predicted_result:
                    predict = val_logits.numpy()
                    for index in range (0, batch_size):
                        predicted_image = predict[index,:,:,:]
                        predicted_image = predicted_image.reshape(256,256,3)
                        predicted_image = (predicted_image*255)
                        
                        ground_truth = output_test.numpy()
                        ground_truth = ground_truth[index,:,:,:]
                        ground_truth = ground_truth.reshape(256,256,3)
                        ground_truth = (ground_truth*255)
                        
                        save_as_png(SAVE_PATH_PREDICTED, frame_test_folder, each_group, f"{start + index}", predicted_image)
                        save_as_png(SAVE_PATH_GROUND_TRUTH, frame_test_folder, each_group, f"{start + index}", ground_truth)
    ssim_loss_avg = np.mean(np.array(ssim_loss_holder))
    mse_loss_avg = np.mean(np.array(mse_loss_holder))
    psnr_loss_avg = np.mean(np.array(psnr_loss_holder))
    print(f"Predict SSIM : {ssim_loss_avg}  {mse_loss_avg}  {psnr_loss_avg}")
    print(count_category)            
#%%
with tf.device('/gpu:3'):
#    train_epoch() 
    test(save_predicted_result=True)
#%%