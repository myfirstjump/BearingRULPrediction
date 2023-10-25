from py_module.config import Configuration
from py_module.data_reader import DataReader
from py_module.data_preprocessing import DataProprocessing
from py_module.data_exploration import DataExploration
from py_module.data_training import DataTraining
from py_module.plot_module import PlotDesign
from py_module.data_evaluation import DataEvaluation


import os

class FEMTOBearingRULPrediction(object):

    def __init__(self):
        self.config_obj = Configuration()
        self.reader_obj = DataReader()
        self.data_preprocessing_obj = DataProprocessing()
        self.data_exploration_obj = DataExploration()
        self.training_obj = DataTraining()
        self.plotting_obj = PlotDesign()
        self.evaluation_obj = DataEvaluation()

    def data_loading(self):
        '''若需要重新建立featured_data，執行第一行'''
        train_data_dict = self.reader_obj.femto_data_loading_and_preprocessing()
        '''若直接讀取既有的featured_data，執行第二行'''
        # train_data_dict = self.reader_obj.femto_data_loading_features_dataframe()
        '''若需要建立Testing feature data，執行第三行'''
        test_data_dict = self.reader_obj.femto_data_loading_and_preprocessing(testing_data_flag=True)
        '''讀取testing feature data 執行第四行'''
        # test_data_dict = self.reader_obj.femto_data_loading_features_dataframe(testing_data_flag=True)

        return train_data_dict, test_data_dict

    def data_training(self, features_dataframe):
        model, history = self.training_obj.femto_bearing_RUL_prediction_training(features_dataframe)

    def model_evaluation(self, data_dict, model_name):
        model_path = os.path.join(self.config_obj.model_folder, model_name)
        y_dict = self.evaluation_obj.femto_bearing_RUL_prediction(data_dict, model_path, testing_data_flag=False)
        self.plotting_obj.plot_femto_RUL_prediction_plot(y_dict)
        

def main_flow():
    ''''''
    main_obj = FEMTOBearingRULPrediction()
    # main_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_dict, test_data_dict = main_obj.data_loading()
    
    '''Training'''
    # main_obj.data_training(train_data_dict)

    '''Evaluation'''
    # model_name = "RNN-model...."
    # main_obj.model_evaluation(train_data_dict, model_name)



if __name__ == "__main__":
    main_flow()

