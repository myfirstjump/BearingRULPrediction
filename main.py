from py_module.config import Configuration
from py_module.data_reader import DataReader
from py_module.data_preprocessing import DataProprocessing
from py_module.data_exploration import DataExploration
# from py_module.learning_definition import LearningDefinition
# from py_module.data_training import DataTraining
# from py_module.plot_module import PlotDesign
# from py_module.data_evaluation import DataEvaluation
# from py_module.data_training_tf18 import DataTrainingTF18

import os

class BearingRULPrediction(object):

    def __init__(self):
        self.config_obj = Configuration()
        self.reader_obj = DataReader()
        self.data_preprocessing_obj = DataProprocessing()
        self.data_exploration_obj = DataExploration()
        # self.learing_def_obj = LearningDefinition()
        # self.training_obj = DataTraining()
        # self.plotting_obj = PlotDesign()
        # self.evaluation_obj = DataEvaluation()
        # self.training_tf18_obj = DataTrainingTF18()

    def data_loading(self):

        


        path_dict = self.reader_obj.read_IMS_bearing_data_path()
        print(path_dict)
        return path_dict

    def data_exploration(self, data):

        self.data_exploration_obj.data_exploration_2008_PHM_Engine_data(data)

    def data_preprocessing(self, data):
        
        data = self.data_preprocessing_obj.data_preprocessing_2008_PHM_Engine_data(data, self.config_obj.features_name)
        data = self.data_preprocessing_obj.features_standardization(data, self.config_obj.standardization_features)

        return data
    
    def learning_define(self, data):
        
        ### PHM 2008 Engine, 

        new_data = self.learning_def_obj.learning_define_2008_PHM_Engine_data(data)        

        return new_data

    def model_training(self, data):

        my_history = self.training_obj.training_2008_PHM_Engine_data(data, epochs=30, load_model = False)
        
        return my_history

    def plotting_function(self, obj):

        self.plotting_obj.learning_curve(obj)

    def data_evaluation(self, test_data):
        
        self.evaluation_obj.data_evaluation_2008_PHM_Engine_data(test_data)


def main_flow():
    
    main_obj = BearingRULPrediction()
    
    data = main_obj.data_loading()


if __name__ == "__main__":
    main_flow()

