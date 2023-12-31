import time
import os
import random
import itertools
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn import model_selection, preprocessing

from py_module.plot_module import PlotDesign
from py_module.config import Configuration
from py_module.learning_definition import LearningDefinition

class DataTraining(object):

    def __init__(self):

        self.plot_obj = PlotDesign()
        self.config_obj = Configuration()
        self.learing_def_obj = LearningDefinition()

    def sys_show_execution_time(method):
        def time_record(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            execution_time = np.round(end_time - start_time, 3)
            print('Running function:', method.__name__, ' cost time:', execution_time, 'seconds.')
            return result
        return time_record

    @sys_show_execution_time
    def femto_bearing_RUL_prediction_training(self, features_dataframe_dict):
        '''
        1.get stationary feature data
        2.輸出standard pickle
        3.將數據加入RUL欄位值
        4.建立預測模型並且訓練
        5.輸出模型檔案(.h5, pickle)
        '''
        '''1'''
        for k in features_dataframe_dict.keys():
            features_dataframe_dict[k] = features_dataframe_dict[k].pct_change(1,).iloc[1:, :] #iloc取1:是因為pct_change後1st row為nan
            features_dataframe_dict[k] = features_dataframe_dict[k].reset_index(drop=True)
            # print("Check exp {} for NAN: {}".format(k, features_dataframe_dict[k].isnull().values.any()))
        
        '''2 sc pickle初次需要建立，存入assets\\models'''
        merged_df = pd.concat(features_dataframe_dict.values(), axis=0)
        sc = preprocessing.StandardScaler()
        allFeatures_norm = sc.fit_transform(merged_df)
        sc_path = os.path.join(self.config_obj.model_folder, 'sc.pkl')
        joblib.dump(sc, sc_path)
        del merged_df
        '''3'''
        dataframe = self.define_and_add_RUL_column(features_dataframe_dict)
        # '''Check if NaN values'''
        # for k in dataframe.keys():
        #     is_NaN = dataframe[k].isnull()
        #     row_has_NaN = is_NaN.any(axis=1)
        #     rows_with_NaN = dataframe[k][row_has_NaN]
        #     print("Check exp {} for NaN: {}".format(k, dataframe[k].isnull().values.any()))
        #     print(rows_with_NaN)
        '''4'''
        epochs=10
        model_comment = "test_output"
        model, history = self.femto_bearing_training_RNN(dataframe, epochs=epochs, model_comment=model_comment)
        self.plot_obj.learning_curve(history)
        return model, history
    
    def define_and_add_RUL_column(self, data_dict, add_upper_bound=True):
        if add_upper_bound:
            RUL_upper_bound = self.config_obj.rul_upper_bound
        else:
            RUL_upper_bound = 999999
        for exp_idx, dataframe in data_dict.items():
            nrow = len(dataframe.index)
            exp_RUL = [i for i in range(0, nrow)][::-1]
            '''設定RUL上限值，因為無上限並不合理'''
            exp_RUL = np.clip(exp_RUL, a_min=0, a_max=RUL_upper_bound)
            RUL = pd.Series(exp_RUL)
            data_dict[exp_idx]['RUL'] = RUL
        return data_dict
    
    def femto_bearing_training_RNN(self, dataframe_dict, epochs, model_comment):
        
        features_num = len(dataframe_dict['Bearing1_1'].columns.values)-1 #-1為刪除RUL欄位
        num_lags = self.config_obj.lag_feature_number

        '''Model Design'''
        model = keras.models.Sequential()
        dropout_rate = 0.3
        model.add(keras.layers.GRU(64, activation='elu', input_shape=(num_lags+1, features_num), return_sequences=True))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=dropout_rate))
        model.add(keras.layers.GRU(64, activation='elu', input_shape=(num_lags+1, features_num), return_sequences=True))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=dropout_rate))
        model.add(keras.layers.GRU(32, activation='elu', input_shape=(num_lags+1, features_num)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=dropout_rate))
        model.add(keras.layers.Dense(16, activation='elu', input_shape=(num_lags+1, features_num)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=dropout_rate))
        model.add(keras.layers.Dense(1))

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.005)
        model.compile(optimizer=optimizer, loss=self.custom_loss_function, metrics=self.custom_loss_function)

        '''Training'''
        training_cnt = 0
        my_history = {'train_loss':[], 'valid_loss':[]}

        '''以k-fold切割數據'''
        exp_num = len(dataframe_dict)
        exp_names = [name for name in dataframe_dict.keys()]

        train_exp, valid_exp = model_selection.train_test_split(exp_names, test_size=0.2)
        
        while True:
            training_cnt += 1
            if training_cnt > epochs:
                break
            print("--------------------------------------------------- RNN Training Epoch {} ---------------------------------------------------".format(training_cnt))
            random.shuffle(train_exp)
            for exp_name in train_exp:
                '''1.建立train/valid data'''
                data_train0 = dataframe_dict[exp_name].copy()
                valid_exp_name = random.choice(valid_exp)
                data_valid0 = dataframe_dict[valid_exp_name].copy()
                '''2.正規化label以外的欄位'''
                features_name = data_train0.columns.values[:-1]
                sc_path = os.path.join(self.config_obj.model_folder, 'sc.pkl')
                sc:preprocessing.StandardScaler = joblib.load(sc_path)
                data_train0[features_name] = sc.transform(data_train0[features_name])
                data_valid0[features_name] = sc.transform(data_valid0[features_name])
                '''3.新增衍生欄位 --> 滯候特徵(lag feature)'''
                data_train = self.learning_define_femto_Bearing_data(data_train0, features_num, num_lags)
                data_valid = self.learning_define_femto_Bearing_data(data_valid0, features_num, num_lags)

                '''4.將X,y分開'''
                train_x = data_train.values[:,:-1]
                train_y = data_train.values[:,-1]
                valid_x = data_valid.values[:,:-1]
                valid_y = data_valid.values[:,-1]

                '''5.轉換為RNN輸入shape'''
                train_x = train_x.reshape((train_x.shape[0], num_lags+1, features_num))
                valid_x = valid_x.reshape((valid_x.shape[0], num_lags+1, features_num))

                '''6.清空RNN cell中的state記憶'''
                model.reset_states()

                history = model.fit(train_x, train_y, epochs=1, batch_size=32, validation_data=(valid_x, valid_y), verbose=1, shuffle=False)

                my_history['train_loss'].append(history.history['loss'][0])
                my_history['valid_loss'].append(history.history['val_loss'][0])
        
        file_time = time.ctime().split()
        file_time = '-'.join(file_time).replace(':', '')
        h5_data_path = os.path.join(self.config_obj.model_folder, '{}-model-{}-{}.h5'.format("RNN", file_time, model_comment))
        model.save(h5_data_path)
        return model, my_history
    



    ### custom loss
    def custom_loss_function(self, y_true, y_pred):

        # print("y_true:", y_true)
        # diff_vec = y_true - y_pred
        # def each_loss_apply(d):
        #     diff_value = tf.cond(d<0, true_fn=lambda: tf.exp(-d/9)-1, false_fn=lambda: tf.exp(d/13)-1)
        #     # diff_value = tf.cond(d<0, true_fn=lambda: -5*d, false_fn=lambda: tf.exp(d/8)-1)
        #     return diff_value
        # diff_vec = tf.map_fn(each_loss_apply, diff_vec)

        '''MSE'''
        squared_difference = tf.square(y_true - y_pred)
        loss = tf.reduce_mean(squared_difference, axis=-1)

        return loss
    
    def learning_define_femto_Bearing_data(self, data, features_num, num_lags):
        '''
        1.獲得lag features
        '''
        unit_data = data
        nrow = len(unit_data.index)
        unit_RUL = unit_data.pop('RUL')
        new_unit = self.pre_timesteps_supervised(unit_data, unit_RUL, n_features=features_num, n_in=num_lags)
        return new_unit
    
    def pre_timesteps_supervised(self, data, target, n_features=5, n_in=1, n_out=1, dropnan=True):

        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()


        for i in range(n_in, 0 ,-1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names

        if dropnan:
            agg.dropna(inplace=True)

        agg['target'] = target.iloc[n_in:, ].values

        return agg


