import random

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

from py_module.config import Configuration
from py_module.plot_module import PlotDesign
from py_module.learning_definition import LearningDefinition
from py_module.data_training import DataTraining

class DataEvaluation(object):

    def __init__(self):
        self.config_obj = Configuration()
        self.plotting_obj = PlotDesign()
        self.learing_def_obj = LearningDefinition()
        self.training_obj = DataTraining()

    def data_evaluation_2008_PHM_Engine_data(self, data):

        h5_path = self.config_obj.keras_model_path
        model = keras.models.load_model(h5_path, compile=False)
        # model.compile(optimizer = keras.optimizers.RMSprop(), loss = self.training_obj.custom_loss_function, metrics = ['mse'])
        model.compile(optimizer = keras.optimizers.RMSprop(), loss = self.training_obj.custom_loss_function)

        def yield_unit_data(data, train_valid_units, epochs):
            cnt = 0
            while cnt < epochs:
                which_unit = random.choice(train_valid_units)
                unit_data = data[data['unit'] == which_unit]
                cnt += 1
                yield which_unit, unit_data
        test_unit_num, test_data = [(test_unit_num, test_data) for (test_unit_num, test_data) in yield_unit_data(data, [i+1 for i in range(self.config_obj.test_engine_number)], 1)][0]

        # def custom_loss_function(y_true, y_pred):
        #     squared_difference = tf.square(y_true - y_pred)
        #     return tf.reduce_mean(squared_difference, axis=-1)

        test_data = self.learing_def_obj.learning_define_2008_PHM_Engine_data(test_data)
        print("以引擎 unit: {} 做為testing data.".format(test_unit_num))

        test_x = test_data.values[:,:-1]
        test_y = test_data.values[:, -1]

        test_x = test_x.reshape((test_x.shape[0], self.config_obj.previous_p_times + 1, self.config_obj.features_num))
        predict_y = model.predict(test_x)

        # plotting
        self.plotting_obj.plot_RUL_prediction(pred_y=predict_y, true_y=test_y, main_unit=test_unit_num)

        # score calculate

        # results = model.evaluate(test_x, test_y, batch_size=None)
        # print("Evaluate Results:", results) # loss
        scores = []
        for test_unit_num in [i+1 for i in range(self.config_obj.test_engine_number)]:
            
            test_data = data[data['unit'] == test_unit_num]
            test_data = self.learing_def_obj.learning_define_2008_PHM_Engine_data(test_data)
            test_x = test_data.values[:,:-1]
            test_y = test_data.values[:, -1]
            test_x = test_x.reshape((test_x.shape[0], self.config_obj.previous_p_times + 1, self.config_obj.features_num))
            results = model.evaluate(test_x, test_y, batch_size=None)
            scores.append(results)
            print('-----> Model has loss {} on engine {}'.format(results, test_unit_num))
        print("Total average score:", np.mean(scores))


