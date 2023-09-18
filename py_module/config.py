import os
import time

class Configuration(object):
    
    def __init__(self):

        ### data source
        self.data_folder = "/data/archive"
        self.exp_1_folder = "1st_test/1st_test"
        self.exp_2_folder = "2nd_test/2nd_test"
        self.exp_3_folder = "3rd_test/4th_test/txt"

        ### data columns name
        self.exp_1_channel_name_list = ["exp_1_ch1", "exp_1_ch2", "exp_1_ch3", "exp_1_ch4", "exp_1_ch5", "exp_1_ch6", "exp_1_ch7", "exp_1_ch8", ]
        self.exp_2_channel_name_list = ["exp_2_ch1", "exp_2_ch2", "exp_2_ch3", "exp_2_ch4", ]
        self.exp_3_channel_name_list = ["exp_3_ch1", "exp_3_ch2", "exp_3_ch3", "exp_3_ch4", ]

        


