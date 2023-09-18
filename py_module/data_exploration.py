import pandas as pd
from py_module.config import Configuration

class DataExploration(object):

    def __init__(self):
        self.config_obj = Configuration()
        pass

    def data_exploration_2008_PHM_Engine_data(self, data):

        # 查看每個引擎init的cycles數量

        for unit in range(1, self.config_obj.train_engine_number + 1):
            # unit_cycle = data.loc[data.unit == unit].time_in_cycles
            # print('Engine Unit:{} with Cycles:{}'.format(unit, unit_cycle))

            unit_data = data.loc[data.unit == unit]
            nrow = len(unit_data.index)

            # print('Engine Unit:{} with Cycles:{}'.format(unit, nrow))