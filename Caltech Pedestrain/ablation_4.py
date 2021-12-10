#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:36:27 2021

@author: navin
"""
import tensorflow as tf
with tf.device('/gpu:1'):
    import cv2
    import warnings
    
    from tensorflow import keras
    from tensorflow.keras import layers
    
    from dataloader_256_256_stack_1 import MultiInputGenerator
    from loss import ssim_loss, squared_distance_loss, psnr_loss
    from loss1 import losses
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    frame_path = '/home/navin/paper_5/DATASET_10/frame/'
    optical_path = '/home/navin/paper_5/DATASET_10/optical_flow/'
    
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
            max_pool = self.conv_pooling(input_tensor)
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
            
        def call(self, input_tensor, frame_conn, optical_conn, model, optical_feed=False, concatenation=False, training=False):
            up_sampling = self.conv_transpose(input_tensor, training=training)
            up_sampling = self.b_n(up_sampling, training=training)
    #        print(f"Up_sampling: ", up_sampling.shape)
            if model == 'main':
                concat_frame_optical = self.concat([up_sampling, frame_conn])
                convolution_block = self.cnn_block1(concat_frame_optical, training=training)
                if optical_feed:
                    mask = self.mask(convolution_block, training=training)
                    mask_mul = self.multiplication([mask, optical_conn])
                    add = self.add([convolution_block, mask_mul])
                    convolution_block = self.cnn_block2(add, training=training)
            elif model == 'optical':
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
            o_pool3 = self.optical_pool3(e_optical3)
    #        ### optical Encoder Block 4
            e_optical4 = self.optical_block4(o_pool3, training=training)
            o_pool4 = self.optical_pool4(e_optical4, training=training)
            ##
            e_optical5 = self.optical_block5(o_pool4, training=training)
            ### Reshape to fit ConvLSTM2D layer
    #        reshape_in = self.reshape_rnn_in(e_optical5)
            ### Recurrent Layer
    #        optical_rnn1 = self.optical_rnn_block1(reshape_in, training=training)
            ###Reshape to fit decoder block 
    #        optical_rnn1 = self.reshape_rnn_out(optical_rnn1)
            ### Optical Decoder Block 4
            d_optical4 = self.decode_block4(e_optical5, e_optical4, e_optical4, 'optical', optical_feed=False, concatenation=True, training=training)
    #        ### Optical Decoder Block 3
            d_optical3 = self.decode_block3(d_optical4, e_optical3, e_optical3, 'optical', optical_feed=False, concatenation=True, training=training)
            ### Optical Decoder Block 2
            d_optical2 = self.decode_block2(d_optical3, e_optical2, e_optical2, 'optical', optical_feed=False, concatenation=True, training=training)
            ### Optical Decoder Block 1
            d_optical1 = self.decode_block1(d_optical2, e_optical1, e_optical1, 'optical', optical_feed=False, concatenation=True, training=training)
            
            # output layer
            output = self.out_layer(d_optical1)
            return([output, e_optical5, d_optical4, d_optical3, d_optical2, d_optical1])
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
            
            optical_output, optical_rnn1, d_optical4, d_optical3, d_optical2, d_optical1= self.optical_model(inputs[1], training=training1)
            #Reshape to fit to ConvLSTM
    #        reshape_in = self.reshape_rnn_in(frame5)
            #ConvLSTM to learn timeseries
    #        frame_rnn1 = self.frame_rnn_block1(reshape_in, training=training)
            ###Reshape to fit decoder block 
    #        frame_rnn1 = self.reshape_rnn_out(frame_rnn1)
            
            # Model Decoder --> concat frame and optical
            decoder4 = self.decode_block4(frame5, frame4, frame4, 'main',  optical_feed=False, concatenation=True, training=training)
            decoder3 = self.decode_block3(decoder4, frame3, frame4, 'main',  optical_feed=False, concatenation=True, training=training)
            decoder2 = self.decode_block2(decoder3, frame2, frame4, 'main',  optical_feed=False, concatenation=True, training=training)
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
    def train_optical(x_frame, x_optical, output, output_optical):
        with tf.GradientTape() as tape1:
            logits1,_,_,_,_,_ = model_optical(x_optical, training=True)
            loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow, loss = compute_loss_optical(x_optical, output_optical, logits1)
        grads1 = tape1.gradient(loss , model_optical.trainable_variables)
        optimizer_optical.apply_gradients(zip(grads1, model_optical.trainable_variables))
        return loss
    #%%
    data_loader = MultiInputGenerator(frame_path, optical_path)
    test_data_loader = MultiInputGenerator(frame_path, optical_path)
    #%%    
    def train_epoch(model, epochs = 100, batch_size = 8):
        past_sequence = 4
        prediction_step = 1
        total_sample = 180000
        app = []
        mse_h = []
        psnr_h = []
    #    start = 0
        for epoch in range (0, epochs):
            for start in range(0, total_sample - past_sequence - batch_size, batch_size):
                x_frame, x_optical, output, output_optical = data_loader.get_data(start, past_sequence + 1, prediction_step, batch_size)
                loss_value_opt = train_optical(x_frame, x_optical, output, output_optical)
                loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow, loss_value = train_main(x_frame, x_optical, output, output_optical)
                if start % 8000 == 0:
                    print(f"Epochs {epoch}  {start} Training Loss: {float(loss_value)} {loss_value_opt}")
                    print(loss_value_gdl, loss_value_lse, loss_value_mae, loss_value_ssim, loss_value_flow)
    #            if start % 5000 == 0: 
            test_loss = 0  
            mse_loss = 0
            p_loss = 0
            for k in range(180000, 249000 - past_sequence - batch_size, batch_size):
    #            print(k)
                x_frame_test, x_optical_test, output_test, output_optical_test = data_loader.get_data(k, past_sequence + 1, prediction_step, batch_size)
                val_logits = model([x_frame_test, x_optical_test], training=False)
                test_loss += ssim_loss(output_test, val_logits)
                mse_loss += squared_distance_loss(output_test, val_logits)
                p_loss += psnr_loss(output_test, val_logits)
                app.append(test_loss)
                mse_h.append(mse_loss)
                psnr_h.append(p_loss)
            test_loss_avg = (test_loss/(249000 - past_sequence - batch_size - 180000)*batch_size)
            mse_loss_avg = (mse_loss/(249000 - past_sequence - batch_size - 180000)*batch_size)
            p_loss_avg = (p_loss/(249000 - past_sequence - batch_size - 180000)*batch_size)
            print(f"Predict SSIM : {test_loss_avg}  {mse_loss_avg}  {p_loss_avg}")
            for k in range(185150, 185400 - past_sequence - batch_size, batch_size):
                x_frame_test, x_optical_test, output_test, output_optical_test = data_loader.get_data(k, past_sequence + 1, prediction_step, batch_size)
                val_logits = model([x_frame_test, x_optical_test], training=False)
                predict = val_logits.numpy()
                for index in range (0,8):
                    out = predict[index,:,:,:]
                    out = out.reshape(256,256,3)
                    out = (out*255)
                    predict1 = output_test.numpy()
                    out1 = predict1[index,:,:,:]
                    out1 = out1.reshape(256,256,3)
                    out1 = (out1*255)
    #            ss = ssim_loss(output_test, val_logits)
    #            app.append(ss)
    #    return [app, mse_h, psnr_h]
    #                o = cv2.hconcat([out,out1])
                    cv2.imwrite(f"/home/navin/paper_5/dataset/Output/ablation_3_1/{epoch}_{k+index}.png", out)
                    cv2.imwrite(f"/home/navin/paper_5/dataset/Output/ablation_3_2/{epoch}_{k+index}.png", out1)
    #        for k in range(247600, 247950 - past_sequence - batch_size, batch_size):
    #            x_frame_test, x_optical_test, output_test, output_optical_test = data_loader.get_data(k, past_sequence + 1, prediction_step, batch_size)
    #            val_logits = model([x_frame_test, x_optical_test], training=False)
    #            predict = val_logits.numpy()
    #            for index in range (0,8):
    #                out = predict[index,:,:,:]
    #                out = out.reshape(256,256,3)
    #                out = (out*255)
    #                predict1 = output_test.numpy()
    #                out1 = predict1[index,:,:,:]
    #                out1 = out1.reshape(256,256,3)
    #                out1 = (out1*255)
    #            ss = ssim_loss(output_test, val_logits)
    #            app.append(ss)
    #    return [app, mse_h, psnr_h]
    #                o = cv2.hconcat([out,out1])
    #                cv2.imwrite(f"/home/navin/paper_5/dataset/Output/result1_4_23/epoch_{k+index}.png", out)
    #                cv2.imwrite(f"/home/navin/paper_5/dataset/Output/result1_1_4_23/epoch_{k+index}.png", out1)
    #                model.save_weights(f"/home/navin/paper_5/dataset/Output/model_pro_1.h5")   
    #                model_optical.save_weights(f"/home/navin/paper_5/dataset/Output/model_optical_1.h5") 
    #                val_logits_optical,_,_,_,_,_ = model_optical(x_optical_test, training=False)
    #                
    #                val_logits_optical = val_logits_optical.numpy()
    #                optical_predict = val_logits_optical[0,:,:,:]
    #                optical_predict = optical_predict.reshape(256,256,3)
    #                optical_predict = optical_predict*255
    #                
    #                optical_true = output_optical_test.numpy()
    #                optical_true = optical_true[0,:,:,:]
    #                optical_true = optical_true.reshape(256,256,3)
    #                optical_true = optical_true*255
    #                oo = cv2.hconcat([optical_predict, optical_true])
    #                cv2.imwrite(f"/home/navin/paper_5/dataset/Output/G_13/op/{epoch}_{start}_{k}.png", oo)
    #        if epoch >=12:
    #            for k in range(20001, 20500 - past_sequence - batch_size):
    #                x_frame_test, x_optical_test, output_test,_ = data_loader.get_data(k, past_sequence + 1, prediction_step, batch_size)
    #                val_logits = model([x_frame_test, x_optical_test], training=False)
    #                predict = val_logits.numpy()
    #                out = predict[0,:,:,:]
    #                out = out.reshape(256,256,3)
    #                cv2.imwrite(f"/home/navin/paper_5/dataset/Output/G_1/{epoch}/{k}.png",(out*127.5)+127.5)
        return [app, mse_h, psnr_h]
    #%%

    xxxx = train_epoch(model) 
    #%%
#import numpy as np
#import matplotlib.pyplot as plt
#yy = np.array(xxxx)
#np.mean(yy)
#len(yy[0])
##%%
#SS = yy[0][100:200]
##SS = np.sort(SS)
#SS = SS[SS<=0.09]
#plt.plot(SS)
##np.max(SS)
#1-np.mean(SS)
##np.std(SS)
##%%
#me = yy[1][0:1000]
##me = np.sort(me)
#me = me[me<=0.006]
#plt.plot(me)
#np.mean(me)
##np.max(me)
##%%
#me = yy[2][0:6000]
##me = np.sort(me)
#me = me[me>=30]
#plt.plot(me)
#np.mean(me)
##np.max(me)
#%%