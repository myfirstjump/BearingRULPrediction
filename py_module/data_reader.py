from py_module.config import Configuration

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.fftpack import fft

class DataReader(object):

    def __init__(self):
        self.config_obj = Configuration()

    def femto_data_loading_and_preprocessing(self, testing_data_flag=False):
        
        '''
        1.讀取FEMTO Bearing資料集(只讀取acc資料，不讀取Temperature資料)
        2.進行特徵工程
        3.存入以實驗名稱為Key的字典
        data_dict = {"Bearing1_1": dataframe, "Bearing1_2": dataframe, "Bearing2_1": dataframe, ...}
        '''
        train_data_folder = self.config_obj.training_data_folder
        test_data_folder = self.config_obj.testing_data_folder

        if testing_data_flag:
            new_data_dict = {
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
            new_data_dict = {
                "Bearing1_1":[],
                "Bearing1_2":[],
                "Bearing2_1":[],
                "Bearing2_2":[],
                "Bearing3_1":[],
                "Bearing3_2":[],
            }
        
        freq_x_bin = ([0, 200], [200, 1000])
        freq_y_bin = ([0, 200], [200, 1000])

        new_dataframe = {k:None for k in new_data_dict.keys()}

        '''1'''
        if testing_data_flag:
            target_folder = test_data_folder
        else:
            target_folder = train_data_folder
        for idx, sub_folder in enumerate(os.listdir(target_folder)):
            sub_folder_path = os.path.join(target_folder, sub_folder)
            data_path = [os.path.join(sub_folder_path, name) for name in os.listdir(sub_folder_path) if (os.path.isfile(os.path.join(sub_folder_path, name)) and ("acc" in name))]
            print("Training data {} folder: {}, total {} files.".format(idx, sub_folder, len(data_path)))

            for each_idx, each_path in enumerate(data_path):
                print(each_idx, each_path)
                '''
                vibration data is on col4, col5(note that if the temperature data is needed, the seperate sign is ';')
                '''
                if sub_folder in ["Bearing1_4"]: #Bearing1_4的分隔符為;不是,
                    data = pd.read_csv(each_path, header=None, sep=";", usecols=[4,5], names=['x', 'y'])
                else:
                    data = pd.read_csv(each_path, header=None, usecols=[4,5], names=['x', 'y'])
                '''2'''
                feature_x = self.vibration_features_transformation(data['x'])
                feature_y = self.vibration_features_transformation(data['y'])

                my_fft_x, freq_x = self.fourier_transformation(data['x'])
                magnitude_x = self.avg_magnitude_by_bins(my_fft_x, freq_x, freq_x_bin)
                my_fft_y, freq_y = self.fourier_transformation(data['y'])
                magnitude_y = self.avg_magnitude_by_bins(my_fft_y, freq_y, freq_y_bin)

                fft_features = []
                for fea in magnitude_x:
                    fft_features.append(fea)
                for fea in magnitude_y:
                    fft_features.append(fea)
                
                each_new_data_dict = {
                    'X_mean':0.0, 
                    'X_std' :0.0,
                    'X_rms':0.0, 
                    'X_crestf':0.0, 
                    'X_skew':0.0, 
                    'X_kurtosis':0.0, 
                    'X_max':0.0, 
                    'X_min':0.0, 
                    'X_p2p':0.0, 
                    'Y_mean':0.0, 
                    'Y_std':0.0, 
                    'Y_rms':0.0, 
                    'Y_crestf':0.0, 
                    'Y_skew':0.0, 
                    'Y_kurtosis':0.0, 
                    'Y_max':0.0,
                    'Y_min':0.0,
                    'Y_p2p':0.0,
                    'freqX_1':0.0,
                    'freqY_1':0.0,
                }

                each_new_data_dict['X_mean'] = feature_x['mean']
                each_new_data_dict['X_std'] = feature_x['std']
                each_new_data_dict['X_rms'] = feature_x['rms']
                each_new_data_dict['X_crestf'] = feature_x['crestf']
                each_new_data_dict['X_skew'] = feature_x['skew']
                each_new_data_dict['X_kurtosis'] = feature_x['kurtosis']
                each_new_data_dict['X_max'] = feature_x['max']
                each_new_data_dict['X_min'] = feature_x['min']
                each_new_data_dict['X_p2p'] = feature_x['p2p']

                each_new_data_dict['Y_mean'] = feature_y['mean']
                each_new_data_dict['Y_std'] = feature_y['std']
                each_new_data_dict['Y_rms'] = feature_y['rms']
                each_new_data_dict['Y_crestf'] = feature_y['crestf']
                each_new_data_dict['Y_skew'] = feature_y['skew']
                each_new_data_dict['Y_kurtosis'] = feature_y['kurtosis']
                each_new_data_dict['Y_max'] = feature_y['max']
                each_new_data_dict['Y_min'] = feature_y['min']
                each_new_data_dict['Y_p2p'] = feature_y['p2p']

                each_new_data_dict['freqX_1'] = fft_features[0]
                each_new_data_dict['freqY_1'] = fft_features[2]

                new_data_dict[sub_folder].append(each_new_data_dict)

        for each_exp in new_dataframe.keys():
            new_dataframe[each_exp] = pd.DataFrame.from_dict(new_data_dict[each_exp])
            new_dataframe[each_exp].to_csv(os.path.join(self.config_obj.featured_data_folder, 'feature_data_{}.csv'.format(each_exp)), index=False, sep=',')
        
        return new_dataframe
    
    def femto_data_loading_features_dataframe(self, testing_data_flag=False):
        '''
        讀取femto_data_loading_and_preprocessing建立並儲存的dataframe
        '''
        folder_path = self.config_obj.featured_data_folder
        data_path_list = os.listdir(folder_path)
        file_name = []
        file_path_list = []

        train_exp = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]
        test_exp = ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7", "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7", "Bearing3_3"]

        for each_data_name in data_path_list:
            file_path = os.path.join(folder_path, each_data_name)
            if os.path.isfile(file_path):
                each_data_name = each_data_name.split(".csv")[0].split('feature_data_')[1] # 將feature_data_exp3_Ch4.csv轉為 exp3_Ch4
                file_name.append(each_data_name)
                file_path_list.append(file_path)
        
        if testing_data_flag:
            '''若要load train，把test exp drop，若是要load test則相反'''
            for exp in train_exp:
                idx = file_name.index(exp)
                file_name.pop(idx)
                file_path_list.pop(idx)
        else:
            for exp in test_exp:
                idx = file_name.index(exp)
                file_name.pop(idx)
                file_path_list.pop(idx)
        
        data_dict = {}
        for idx, each_file_path in enumerate(file_path_list):
            print("讀取資料檔案{}".format(each_file_path))
            data_dict[file_name[idx]] = pd.read_csv(each_file_path, sep=',', header=0)
        return data_dict
    
    def vibration_features_transformation(self, data):

        new_data = {
            'mean':0.0,
            'std':0.0,
            'rms':0.0,
            'crestf':0.0,
            'skew':0.0,
            'kurtosis':0.0,
            'max':0.0,
            'min':0.0,
            'p2p':0.0,
        }
        new_data['mean'] = np.mean(data)
        new_data['std'] = np.std(data)
        new_data['rms'] = np.sqrt(np.mean(data**2))
        new_data['crestf'] = np.max(data) / new_data['rms']
        new_data['skew'] = stats.skew(data)
        new_data['kurtosis'] = stats.kurtosis(data)
        new_data['max'] = np.max(data)
        new_data['min'] = np.min(data)
        new_data['p2p'] = new_data['max']-new_data['min']

        return new_data

    def fourier_transformation(self, data):
        '''
        Apply fourier transformation to data(raw signal)
        '''

        data_ary = np.array(data)
        n = len(data_ary)
        n2 = int(n/2)
        Fs = 25600 # 採樣頻率
        T = 1/Fs #週期
        dF = Fs/n
        t = np.linspace(0, n-1, n) * T
        freq = np.linspace(0, n2-1, n2) * dF

        rpm = 2000
        my_fft = abs(fft(data_ary)) * 2 / len(data_ary)
        my_fft = my_fft[0:n2]
        '''fft plot to check appropriate freq interval'''
        # plt.subplot(2,1,1)
        # plt.plot(t, data_ary)
        # plt.title('Time Domain')

        # plt.subplot(2,1,2)
        # plt.plot(freq, my_fft)
        # plt.title('Frequency Domain')
        # plt.tight_layout()

        # plt.show()
        return my_fft, freq
    
    def dataframe_moving_average(self, dataframe, ma_features, ma_interval):
        for each_feature in ma_features:
            dataframe[each_feature] = dataframe[each_feature].rolling(window=ma_interval).mean()
        return dataframe
    
    def avg_magnitude_by_bins(self, fft, freq, bins):
        '''
        將freq依據bins的範圍分割，把fft數據依區間處理(取max、...)'''
        magnitude = []
        for each_bin in bins:
            index_ = np.where( (freq >= each_bin[0]) & (freq <= each_bin[1]) )
            magnitude.append(np.max(fft[index_]))
        
        return magnitude
                

        
                   
