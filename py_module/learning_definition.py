from py_module.config import Configuration

import pandas as pd
import numpy as np
import random
# from sklearn import model_selection

class LearningDefinition(object):

    def __init__(self):
        self.config_obj = Configuration()

    def learning_define_2008_PHM_Engine_data(self, data):

        unit_data = data
        nrow = len(unit_data.index)
        unit_data.pop('unit')
        # print('Engine Unit:{} with cycles:{}'.format(unit, nrow))
        unit_RUL = unit_data.pop('RUL')
        new_unit = self.pre_timesteps_supervised(unit_data, unit_RUL, n_features=self.config_obj.features_num, n_in=self.config_obj.previous_p_times)

        return new_unit

    def pre_timesteps_supervised(self, data, target, n_features=5, n_in=1, n_out=1, dropnan=True):

        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()


        for i in range(n_in, 0 ,-1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names

        if dropnan:
            agg.dropna(inplace=True)

        agg['target'] = target.iloc[n_in:, ].values

        return agg