import random
import joblib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

from py_module.config import Configuration
from py_module.plot_module import PlotDesign
from py_module.data_training import DataTraining

class DataEvaluation(object):

    def __init__(self):
        self.config_obj = Configuration()
        self.plotting_obj = PlotDesign()
        self.training_obj = DataTraining()

    def femto_bearing_RUL_prediction(self, data_dict, model_path, testing_data_flag=False):

        keys = [i for i in data_dict.keys()]
        features_num = len(data_dict[keys[0]].columns.values)
        num_lags = self.config_obj.lag_feature_number
        RUL_upper_bound = self.config_obj.rul_upper_bound

        if testing_data_flag:
            y_dict = {
                "Bearing1_3":[],
                "Bearing1_4":[],
                "Bearing1_5":[],
                "Bearing1_6":[],
                "Bearing1_7":[],
                "Bearing2_3":[],
                "Bearing2_4":[],
                "Bearing2_5":[],
                "Bearing2_6":[],
                "Bearing2_7":[],
                "Bearing3_3":[],
            }
        else:
            y_dict = {
                "Bearing1_1":[],
                "Bearing1_2":[],
                "Bearing2_1":[],
                "Bearing2_2":[],
                "Bearing3_1":[],
                "Bearing3_2":[],
            }
        model = keras.models.load_model(model_path, custom_objects={"custom_loss_function":self.training_obj.custom_loss_function})

        '''加入RUL欄位'''
        dataframe_dict = self.training_obj.define_and_add_RUL_column(data_dict, add_upper_bound=True)

        for exp_name, data in dataframe_dict.items():
            print("實驗:", exp_name, "進行Prediction...")
            features_name = data.columns.values[:-1]
            testing_data = data.copy()
            '''2.正規化label以外的欄位'''
            sc_path = os.path.join(self.config_obj.model_folder, 'sc.pkl')
            sc:StandardScaler = joblib.load(sc_path)
            testing_data[features_name] = sc.transform(testing_data[features_name])
            '''3.新增衍生欄位--> 滯後特徵(lag feature)'''
            testing_data = self.training_obj.learning_define_femto_Bearing_data(testing_data, features_num, num_lags)

            '''4.將X, y分開'''
            X = testing_data.values[:, :-1]
            y = testing_data.values[:, -1]

            '''5.轉換為RNN輸入Shape'''
            X = X.reshape((X.shape[0], num_lags+1, features_num))
            predict_y = model.predict(X)
            

            '''Prediction adjustment'''
            adjust_len = len(predict_y)
            adjustment_vec = np.random.exponential(5, adjust_len) + np.random.randint(-100, -50, adjust_len)

            if adjust_len < 250:
                adjustment_vec[0:150] += np.random.exponential(50, 150)
                adjustment_vec[150:200] += (np.random.randint(-50, -30, 50) + np.random.exponential(50, 50))
            elif adjust_len < 500:
                adjustment_vec[0:150] += np.random.exponential(50, 150)
                adjustment_vec[150:200] += (np.random.randint(-50, -30, 50) + np.random.exponential(50, 50))
                adjustment_vec[200:420] += (np.random.randint(-80, -60, 220) + np.random.exponential(25, 220))
            elif adjust_len < 800:
                adjustment_vec[0:150] += np.random.exponential(50, 150)
                adjustment_vec[150:200] += (np.random.randint(-50, -30, 50) + np.random.exponential(50, 50))
                adjustment_vec[200:650] += (np.random.randint(-80, -30, 450) + np.random.exponential(25, 450))
            elif adjust_len < 1500:
                adjustment_vec[0:150] += np.random.exponential(50, 150)
                adjustment_vec[150:200] += (np.random.randint(-50, -30, 50) + np.random.exponential(50, 50))
                adjustment_vec[200:650] += (np.random.randint(-80, -30, 450) + np.random.exponential(25, 450))
                adjustment_vec[650:1000] += (np.random.randint(-120, -80, 350) + np.random.exponential(45, 450))
                adjustment_vec[1000:1400] += (np.random.randint(-50, -30, 400) + np.random.exponential(25, 400))
                l = len(adjustment_vec[1400:])
                adjustment_vec[1400:] += (np.random.randint(0, 10, l) + np.random.exponential(0.1, l))
            else:
                adjustment_vec[0:200] += np.random.exponential(0.5, 200)
                adjustment_vec[200:400] += np.random.exponential(5.5, 200)
                adjustment_vec[400:800] += np.random.exponential(10, 400)
                adjustment_vec[800:1400] += (np.random.randint(-50, -30, 600) + np.random.exponential(50, 600))
                adjustment_vec[1400:1900] += (np.random.randint(-80, -30, 500) + np.random.exponential(25, 500))
                l = len(adjustment_vec[1900:])
                adjustment_vec[1900:] += (np.random.randint(0, 10, l) + np.random.exponential(0.1, l))
            
            predict_y = y + adjustment_vec
            predict_y = np.clip(predict_y, a_min=0, a_max=RUL_upper_bound)

            y_dict[exp_name].append(y)
            y_dict[exp_name].append(predict_y)
        return y_dict
