import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler

from py_module.config import Configuration

class DataProprocessing(object):

    def __init__(self):
        self.config_obj = Configuration()
        

    def data_preprocessing_2008_PHM_Engine_data(self, data, new_col_name):

        data = self.data_col_rename(data, new_col_name)
        data = data.drop(labels=['sensor_22', 'sensor_23'], axis='columns')
        data = self.define_and_add_RUL_column(data)
        data = self.clip_variables(data, variable='RUL', max_=130, min_=0)

        return data

    def data_col_rename(self, data, new_col_name):
        
        # data = data.rename(columns=new_col_name)
        data.columns = new_col_name

        return data

    def data_col_remove(self, data, rm_col_name):
        
        data = data.drop(rm_col_name, axis=1)
    def define_and_add_RUL_column(self, data):

        """
        Function:
            定義2008PHM引擎training資料集的supervised learning模式，新增RUL欄位。
            定義方式為cycle的反序列，比如說一個引擎資料有1~200個cycles，那個RUL的序列即為199, 198, 197, ..., 0。
        Input:
            Training Data
        Output:
            新增一欄位的Training Data
        """
        RUL_list = []

        for unit in range(1, self.config_obj.train_engine_number + 1):
            unit_data = data.loc[data.unit == unit]
            nrow = len(unit_data.index)
            unit_RUL = [i for i in range(0, nrow)][::-1]
            RUL_list = RUL_list + unit_RUL

        RUL = pd.Series(RUL_list)
        data['RUL'] = RUL

        return data

    def features_standardization(self, data, features_str):
        
        scaler = StandardScaler()
        data[features_str] = scaler.fit_transform(data[features_str])

        return data

    def clip_variables(self, data, variable, max_, min_):
        
        series = data[variable]
        new_series = pd.Series([max(min(x, max_), min_) for x in series])

        data[variable] = new_series

        return data