import os
import time

class Configuration(object):
    
    '''參數設定'''
    def __init__(self):

        '''訓練資料路徑'''
        self.training_data_folder = "D:\\myfirstjump_datasets\\FEMTO-ST(PHM_DATA_CHALLENGE_2012)\\phm-ieee-2012-data-challenge-dataset-master\\Learning_set"
        self.testing_data_folder = "D:\\myfirstjump_datasets\\FEMTO-ST(PHM_DATA_CHALLENGE_2012)\\phm-ieee-2012-data-challenge-dataset-master\\Full_Test_Set"

        '''執行檔路徑'''
        working_dir = os.getcwd() #返還main.py檔案資料夾
        self.featured_data_folder = os.path.join(working_dir, "assets\\feature_data")
        self.model_folder = os.path.join(working_dir, "assets\\models")

        '''訓練參數'''
        self.lag_feature_number = 15
        

        


