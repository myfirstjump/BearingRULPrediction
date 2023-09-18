import pandas as pd
import numpy as np

import tensorflow as tf

class DataTrainingTF18(object):

    def __init__(self):
        pass

    def training_PHM_2008_Engine_data(self, data, model_string):

        '''
        RNN的內建類別：BasicRNNCell, BasicLSTMCell
        類別方法： call() 進行RNN單步計算
        類別屬性： state_size, output_size
        '''
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
        print("lstm_cell.state_size:", lstm_cell.state_size)

        inputs = tf.placeholder(np.float32, shape=(32, 100))
        h0 = lstm_cell.zero_state(32, np.float32) # 全為0的initial state
        output, h1 = lstm_cell.call(inputs, h0)

        print("h1.h:", h1.h)
        print("h1.c:", h1.c)


    #     if model_string == 'RNN':

class CallBack(tf.keras.callbacks.Callback):

    # Each epoch end, will call the method on_epoch_end
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.98):
            print('Reached enough accuracy so stop training...')
            self.model.stop_training = True
