import copy
import warnings

from preprocessing.Data import Data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, nbinom

import utils.modelUtils as utils
from model_station.CombinedModelStation import CombinedModelStation
from model_station.ModelGlobal import ModelGlobal
from model_station.ModelStations import ModelStations


# warnings.filterwarnings('error')

class EvaluateModel(object):
    def __init__(self, env, reduction_method, prediction_method, red_dim,
                 **kwargs):
        self.hparam = {
            'load_red': False,          #load reduction method
            'is_model_station': True,   #local/global model
            'is_combined': False,       #ensemble model param in [0,..,8]
            'norm': False,              #normalize objectives
            'load': True,               #load model
            'hours': [],                #hours to consider as features
            'var':False,                #learn a variance predictor
            'zero_prob':False,          #learn a predictor that estimate the 0 probability
            'log': False,               #log transform the objectives
            'mean': False,              #
            'red': {},                  #reduction hparam
            'pred': {},                 #prediction hparam
            'second_pred': {            #variance/zero_prob hparam
                'pred': 'linear'
            }
        }
        self.hparam.update(kwargs)
        # self.is_model_station = is_model_station
        if self.hparam['is_combined']:
            self.mod = CombinedModelStation(env, **self.hparam)
        elif self.hparam['is_model_station']:
            self.mod = ModelStations(env, reduction_method, prediction_method, dim=red_dim,
                                     **self.hparam)
        else:
            self.mod = ModelGlobal(env, reduction_method, prediction_method, **self.hparam)
        self.pred = None
        self.var = None

    def load_or_train(self, data):
        self.mod.load_or_train(data)

    def train(self, data, **kwargs):
        self.pred = None
        self.mod.train(data, **kwargs)

    def save(self):
        self.mod.save()

    def train_variance(self, data, **kwargs):
        self.mod.train_variance(data, **kwargs)

    def train_zero_prob(self, data, **kwargs):
        self.mod.train_zero_prob(data, **kwargs)

    def errors(self, x):
        self.mod.load()
        if self.pred is None:
            self.pred = self.mod.predict(x)
        print('pred_mean', self.pred.mean())
        real = self.mod.get_y(x.get_miniOD(self.hparam['hours']))
        print('real_mean', real.as_matrix().mean())
        self.mod.reset()
        return self.pred - real

    def compute_err(self, x, fct, load=True, station=None, axis=None):
        if load: self.mod.load()
        # if self.pred is None:
        self.pred = self.mod.predict(x)
        if self.var is None and self.hparam['zero_prob']:
            self.var = self.mod.zero_prob(x)
        if load: self.mod.reset()
        y = self.mod.get_y(x.get_miniOD(self.hparam['hours'])).as_matrix()
        if station:
            if self.var:
                return fct(y[:, station], self.pred[:, station], self.var[:, station])
            else:
                return fct(y[:, station], self.pred[:, station], None)
        return fct(y, self.pred, self.var, axis=axis)

    def compute_errors(self, x, fcts_names_, load=True, station=None, axis=None):
        """
        computes errors can computes several errors
        :param x: data tyo predict
        :param fcts_names_: error functions or [mae,rmse,r2,llP,llZI,llNB,rmse_norm,mape,mpe,dev]
        :param load: load models
        :param station: none or station number/names
        :param axis: axis on which aggregate the error
        :return: error numpy array
        """
        s = {'mae': utils.mae,
             'rmse': utils.rmse,
             'r2': utils.r_squared,
             'llP': self.log_likelyhood_poisson,
             'llZI': self.log_likelyhood_zero_inf,
             'llNB': self.log_likelyhood_negative_binomial,
             'rmse_per': utils.rmse_per,
             'rmse_norm': utils.rmse_per,
             'rmsle': utils.rmsle,
             'mape': utils.mape,
             'mpe': utils.mpe,
             'mae3h': utils.ma3e,
             'rmse3h': utils.rms3e,
             'dev': utils.deviation
             }
        fcts = [i if i not in s else s[i] for i in fcts_names_]
        s = self.compute_err(x, fcts[0], load, station, axis=axis).shape
        s = (len(fcts),) + s
        res = np.zeros(s)
        for i, f in enumerate(fcts):
            res[i] = self.compute_err(x, f, load, station, axis=axis)
        return res

    def log_likelyhood_zero_inf(self, y, x, v, station=None, axis=0, var=True):
        try:
            if var :
                self.mod.load()
                print(self.mod.name, '0inf')
                # if self.pred is None:
                self.pred = x
                # if self.var is None:
                self.var = v
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    l = (self.var + self.pred ** 2) / (self.pred) - 1
                l[np.isnan(l)] = 0
                l[l < 0] = 0
                l[self.pred == 0] = 0
                l = utils.maxi(l, self.pred)
                l = utils.maxi(l, self.var)
                l = utils.mini(l, 1000)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pi = 1 - self.pred / l
                pi[l == 0] = 1
                pi[l == 1000] = 1
                pi[pi < 0] = 0
            else:
                self.mod.load()
                print(self.mod.name, '0inf')
                # if self.pred is None:
                self.pred = x
                self.zero_c = v
                self.zero_c = utils.maxi(v, np.exp(-x))
                l = self.pred
                for _ in range(10):
                    expla = utils.maxi(np.exp(-l), 0.000001)
                    pi = (self.zero_c - expla) / (1 - expla)
                    pi = utils.mini(utils.maxi(pi, 0.0001), 0.9999)
                    l = self.pred / (1 - pi)
            real = y
            res = poisson.pmf(real, l)
            res *= (1 - pi)  # * (l ** real) * np.exp(-l) / utils.maxi(1, factorial(real))
            res[pi == 1] = 1
            res[res==0]=0.01
            # utils.savecsv(real[pi == 1][real[pi == 1] >= 10])
            res[real == 0] += pi[real == 0]
            res = np.nanmean(np.log(res), axis=axis)
            self.mod.reset()
            return res
        except ValueError:
            self.pred = None
            self.var = None
            return self.log_likelyhood_zero_inf(y, x, v, station, axis, var)

    def log_likelyhood(self, x:Data, distrib, station=None, axis=None):
        y = x.get_miniOD([])[x.get_stations_col(None)]
        self.mod.load()
        xx = self.mod.predict(x)
        if distrib == 'P':
            self.mod.reset()
            return self.log_likelyhood_poisson(y, xx, None, station=station, axis=axis)
        if distrib == 'NB':
            v=self.mod.variance(x)
            self.mod.reset()
            return self.log_likelyhood_negative_binomial(y, xx, v, station=station, axis=axis)
        if distrib == 'ZI':
            # v=self.mod.zero_prob(x)
            v=self.mod.variance(x)
            self.mod.reset()
            return self.log_likelyhood_zero_inf(y, xx, v, station=station, axis=axis, var=True)

    def log_likelyhood_poisson(self, y, x, v=None, axis=None, station=None):

        # try:
        # self.mod.load()
        # l = self.mod.predict(x)
        # if isinstance(x, Data):
        #     real = self.mod.get_y(x.get_miniOD(self.hparam['hours'])).as_matrix()
        # elif isinstance(x, pd.DataFrame):
        #     real = x.as_matrix()
        # else:
        real = y
        # if station:
        #     res = poisson.pmf(real[:, station], l[:, station])
        # else:
        res = poisson.pmf(real, x)  # l ** real) * np.exp(-l) / utils.maxi(1, factorial(real))
        res = np.nanmean(np.log(res), axis=axis)
        self.mod.reset()
        return res
        # except ValueError:
        #     self.pred = None
        #     return self.log_likelihood_poisson(x)

    def log_likelyhood_negative_binomial(self, y, x, v=None, station=None, axis=0):
        # try:
        print(self.mod.name, 'NB')
        self.mod.load()
        # if self.pred is None:
        self.pred = x
        # if self.var is None:
        self.var = v
        real = y
        p = self.pred / self.var
        p = utils.mini(p, 0.95)
        r = self.pred * p / (1 - p)
        r = utils.maxi(1, r)

        res = nbinom.pmf(real, n=r, p=p)
        # res = gamma(r + real) / (factorial(real) * gamma(r)) * (p ** r) * ((1 - p) ** real)
        res = np.nanmean(np.log(res), axis=axis)
        self.mod.reset()
        return res
        # except ValueError:
        #     self.pred = None
        #     self.var = None
        #     return self.log_likelyhood_negative_binomial(x)

    def plot(self, x, station=None):
        """
        plot the model prediction
        :param x: data to predict
        :param station: station to predict (defailt None)
        :return: none
        """
        self.mod.load()
        # if self.pred is None:
        self.pred = self.mod.predict(x)
        plt.plot(self.pred[300:600, station], label='predicted demand ' + self.mod.name)

    def residuals(self, x, station=None):
        self.mod.load()
        if self.pred is None:
            self.pred = self.mod.predict(x)
        df = copy.deepcopy(x.get_miniOD(self.hparam['hours']))
        df[x.get_stations_col(2015)] = df[x.get_stations_col(2015)] - self.pred
        self.mod.reset()
        return df
