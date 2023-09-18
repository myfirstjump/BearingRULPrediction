from py_module.config import Configuration

import os
import pandas as pd

class DataReader(object):

    def __init__(self):
        self.config_obj = Configuration()

    def read_csv_data(self, data_path):
        
        data = pd.read_csv(data_path, sep=' ', header=None)

        return data

    def read_IMS_bearing_data_path(self):

        
        ### 讀取folder中的檔案列表
        path_dict = {}
        exp_1_path_list = []
        exp_2_path_list = []
        exp_3_path_list = []
        exp_1_file_folder = os.path.join(self.config_obj.data_folder, self.config_obj.exp_1_folder)
        exp_2_file_folder = os.path.join(self.config_obj.data_folder, self.config_obj.exp_2_folder)
        exp_3_file_folder = os.path.join(self.config_obj.data_folder, self.config_obj.exp_3_folder)

        for exp_idx, file_folder in enumerate([exp_1_file_folder, exp_2_file_folder, exp_3_file_folder]):
            
            for each_path in os.listdir(file_folder):
                if exp_idx == 0:
                    exp_1_path_list.append(each_path)
                elif exp_idx == 1:
                    exp_2_path_list.append(each_path)
                else:
                    exp_3_path_list.append(each_path)
        
        path_dict['exp_1'] = exp_1_path_list
        path_dict['exp_2'] = exp_2_path_list
        path_dict['exp_3'] = exp_3_path_list

        return path_dict
            
