#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:34:09 2018

@author: Apple
"""

import numpy as np
import scipy.io as io
import keras
#from keras import backend as K
#from tensorflow.keras import backend
from tensorflow.keras import backend as K
from EEG_models_authentitation_newuser import ShallowConvNet
from tensorflow.keras.callbacks import Callback, EarlyStopping


datapath = 'C:/Users/Lin Guan Ting/SSVEP_UI/'
SSVEP_l_data = io.loadmat(datapath+'SSVEP_l.mat')
SSVEP_l = SSVEP_l_data['SSVEP_l']
SSVEP_label = io.loadmat(datapath+'SSVEP_label.mat')['label'] # i_subj, i_session, i_stim, i_trial

n_class = 2

#x_train = np.squeeze(SSVEP_l[:,:,np.where(SSVEP_label[:,1]==1)])\
# on github
# user authentitaion: output register / non-register user
# SSVEP_label[:,1] is 'session'
log_matrix = np.zeros((8, 8))
'''
a: registered user
b: new testing user
other: training non-registered user
'''
a_list = [1, 2, 3, 4, 5, 6, 7, 8]
for a in a_list:    
    b_list = a_list.copy()
    b_list.remove(a)
    for b in b_list:
        registered_user = a
        new_user = b
        x_train = np.squeeze(SSVEP_l[:,:,np.where(SSVEP_label[:,1]==1 & (SSVEP_label[:,0]!=new_user))]) # find session=1 (x_train)
        # x_train = np.squeeze(SSVEP_l[:,:,np.where((SSVEP_label[:,1]==1) & (SSVEP_label[:,0]==1))])
        x_train = np.transpose(x_train, axes = [2, 0, 1])
        y_train = np.squeeze(SSVEP_label[np.where(SSVEP_label[:,1]==1 & (SSVEP_label[:,0]!=new_user)),0]) # user/non-user
        for i in range(0,len(y_train)):
            if y_train[i]!=registered_user: # others are non-register user
                y_train[i] = 0
            else: 
                y_train[i] = 1
        y_train_onehot = keras.utils.to_categorical(y_train, 2)
        
        x_test = np.squeeze(SSVEP_l[:,:,np.where(SSVEP_label[:,1]==1 & (SSVEP_label[:,0]==new_user))]) # find session=1 (x_test)
        x_test = np.transpose(x_test, axes = [2, 0, 1])
        y_test = np.squeeze(SSVEP_label[np.where(SSVEP_label[:,1]==1 & (SSVEP_label[:,0]==new_user)),0]) # find the session1's subject number (y_train)
        #y_test_onehot = keras.utils.to_categorical(y_test-1, n_class)
        for i in range(0,len(y_test)):
            if y_test[i]!=registered_user:
                y_test[i] = 0
            else: 
                y_test[i] = 1    
        y_test_onehot = keras.utils.to_categorical(y_test, 2)
        '''
        from sklearn.model_selection import train_test_split
        #y_train_onehot = keras.utils.to_categorical(Y, n_class)
        X = np.transpose(X, axes = [2, 0, 1]) # 'axes':permute
        x_train, y_train, x_test, y_test = train_test_split(X, Y, 
                                                            test_size=0.3, random_state=1)
        
        '''
        # input image dimensions
        
        K.set_image_data_format('channels_first')
        n_trial, n_channel, n_timesamp = x_train.shape
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, n_channel, n_timesamp)
            x_test = x_test.reshape(x_test.shape[0], 1, n_channel, n_timesamp)
            input_shape = (1, n_channel, n_timesamp)
        else:
            x_train = x_train.reshape(x_train.shape[0], n_channel, n_timesamp, 1)
            x_test = x_test.reshape(x_test.shape[0], n_channel, n_timesamp, 1)
            input_shape = (n_channel, n_timesamp, 1)
        
        batch_size = 1920
        n_epoch = 400
        varEarlyStopping = 0
        n_patience = 10
        
        model = ShallowConvNet(input_shape)
        #model = SCCNet(input_shape)
        adam = keras.optimizers.adam()
        model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=adam,
                  metrics=['accuracy'])
        '''
        class AccuracyHistory(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.acc = []
            def on_epoch_end(self, batch, logs={}):
                self.acc.append(logs.get('acc'))
        history = AccuracyHistory()
        #    varEarlyStopping = 0 # 0 -> no early stopping; 1 -> early stopping
        if varEarlyStopping == 0:
            callbacks = [history]
        else:    # For early stopping:
            callbacks = [history, EarlyStopping(monitor='val_loss',patience=n_patience, verbose=1, mode='auto', min_delta=0.001)]
        #    batch_size = np.amax([72, int(np.around(x.shape[0]/8))]);
        #    n_epoch = 100
            '''
        #y_train_onehot = np.squeeze(y_train_onehot)    
        history = model.fit(x_train, y_train_onehot,
              batch_size=batch_size,
              epochs=n_epoch,
              verbose=1,
              validation_split=0.25,
              #validation_data=(x_test, y_test_onehot),
              shuffle=True)
        
        result = model.predict(x_test)
        #print(result)
        score = model.evaluate(x_test, y_test_onehot, verbose = 1)
        print('new user testing score: '+str(score))
        log_matrix[a-1,b-1] = score[1]
        # visualization
        import matplotlib.pyplot as plt
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(str(registered_user)+' subject as registered user authentitation accuracy')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend(['train: session 1', 'val: session 1'])
        plt.show()
        





