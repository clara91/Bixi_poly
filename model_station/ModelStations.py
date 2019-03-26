import time

import config
import numpy as np
import model_station.Prediction as pr
import model_station.Reduction as red
from utils.modelUtils import maxi,mini
from preprocessing.Data import Data
import pandas as pd
import joblib


class ModelStations(object):
    def __init__(self, env, reduction_method, prediction_method, dim=10, **kwargs):
        self.hparam = {
            'var':False,
            'zero_prob':False,
            'norm': False,
            'hours': [],
            'load_red': False,
            'log': False,
            'mean': False,
            'decor': False,
            'red':{},
            'pred':{},
            'second_pred': {
                'pred': 'linear',

               },
        }
        
        self.dim =  dim
        self.hparam.update(kwargs)
        print(reduction_method, prediction_method)
        self.env = env
        self.hours = self.hparam['hours']
        self.load_red = self.hparam['load_red']
        self.reduce = red.get_reduction(reduction_method)(env, dim=dim,
                                                                       log=self.hparam['log'], mean=self.hparam['mean'])
        # print('reduce dimension',self.reduce.dim)
        self.meanPredictor = pr.get_prediction(prediction_method)(dim=self.reduce.dim, **self.hparam['pred'])
        # self.varPredictor = pr.get_prediction('linear')(dim=len(Data(env).get_stations_col(None)),
        #                                                 kwargs=kwargs)
        self.featurePCA=None
        self.sum=0
        self.secondPredictor = pr.get_prediction(self.hparam['second_pred']['pred'])(dim=len(Data(env).get_stations_col(None)),
                                                           kwargs=self.hparam['second_pred'])
        self.name = reduction_method + ' ' + prediction_method

    def train(self, data:Data, t=False, **kwargs):
        self.hparam.update(kwargs)
        data.hour = self.hours
        starttime = time.time()
        learn = data
        if self.load_red:
            try:
                self.reduce.load(add_path=self.env.system)
            except (FileNotFoundError, OSError):
                print('load failed, training redution')
                type(self.reduce).train(self.reduce, learn, **self.hparam['red'])
                type(self.reduce).save(self.reduce,add_path=self.env.system)
        else:
            type(self.reduce).train(self.reduce, learn, **self.hparam['red'])
            print("oiiiiiiiiii")
            print(self.env.system)
            type(self.reduce).save(self.reduce, add_path=self.env.system)
        x = self.reduce

        learn2 = type(x).transform(x, learn)

        if self.hparam['decor']:
            WH = self.get_decor_factors(learn)
        else:
            WH = self.get_factors(learn)

        if self.hours != []:
            learn2 = learn2[np.max(self.hours):]
        self.meanPredictor.train(WH, y=learn2, **self.hparam['pred'])
        if self.reduce.algo=='svdevol':
            n=1-(168*self.hparam['n_week'])/data.get_miniOD([]).shape[0]
            self.train_inv(data.get_partialdata_per(n, 1))
        if self.hparam['zero_prob']:
            self.train_zero_prob(learn,**self.hparam['pred'])
        if self.hparam['var']:
            self.train_variance(learn, **self.hparam['pred'])
        if t: print('train time', self.name, time.time() - starttime)
        return time.time() - starttime

    def train_variance(self, learn, **kwargs):
        if isinstance(learn,Data):
            learn.hour=self.hours
        # train variance on old data
        if self.hparam['decor']:
            WH = self.get_decor_factors(learn)
        else:
            WH = self.get_factors(learn)
        res = self.predict(WH)
        WH = self.get_var_factors(learn)
        e = (maxi(res, 0.01) - self.get_objectives(learn)) ** 2

        self.secondPredictor.train(WH, e, **kwargs)

    def train_zero_prob(self, learn, **kwargs):
        # WH = self.get_var_factors(learn)
        # res = self.predict(WH)
        # e = (maxi(res, 0.01) - self.get_objectives(learn)) ** 2
        obj = self.get_objectives(learn)==0
        WH = self.get_factors(learn)
        self.secondPredictor.train(WH, obj, **kwargs)

    # extract learning information
    def get_all_factors(self, learn):
        if isinstance(learn, Data):
            df = learn.get_miniOD(self.hours)
        else:
            df = learn
        col = [i for i in df.columns.values if 'date' not in i]
        return df[col]

    def get_decor_factors(self, learn:(Data,pd.DataFrame)):
        n=20
        df = self.get_factors(learn).as_matrix()
        if self.featurePCA is None:
            self.sum = df.var(axis=0)
            self.mean = df.mean(axis=0)
            df = (df-df.mean(axis=0))/self.sum
            from sklearn.decomposition.pca import PCA
            self.featurePCA = PCA(n).fit(df)
            joblib.dump(self.featurePCA,config.root_path+'featurePCA')
            np.save(config.root_path+'norm_features_var',self.sum   )
            np.save(config.root_path+'norm_features_mean',self.mean   )

        return pd.DataFrame(self.featurePCA.transform((df-df.mean(axis=0))/self.sum),columns=config.get_decor_var(n))
        # if isinstance(learn, Data):
        #     df = learn.get_miniOD(self.hours)
        # else:
        #     df = learn
        # col = config.get_decor_var(20)
        # return df[col]

    def get_factors(self, learn:(Data,pd.DataFrame)):
        if isinstance(learn, Data):
            df = learn.get_miniOD(self.hours)
        else:
            df = learn
        col = []
        for i in df.columns.values:
            for j in config.learning_var:
                if j==i or j == i[:-1] or j == i[:-2]:
                    col.append(i)
        # if 'Date/Heure' in col:
        #     col.remove('Date/Heure')
        # for i in col:
        #     if ' ind' in i or ' Indicateur'in i or 'Dir. ' in i:
        #         col.remove(i)
        col = np.unique(col)
        return df[col]

    def get_var_factors(self, learn:[Data,pd.DataFrame]):

        if isinstance(learn, Data):
            df = learn.get_miniOD(self.hours)
        else:
            df = learn
        col = []
        for i in df.columns.values:
            for j in config.learning_var:
                if j == i or j == i[:-1] or j == i[:-2]:
                    col.append(i)
        col = np.unique(col)
        return df[col]

    def get_objectives(self, learn:Data):
        miniOD = learn.get_miniOD(self.hours)
        return self.reduce.get_y(miniOD)
        # return miniOD[learn.get_stations_col(2015)]

    def predict(self, x, t=False):
        starttime = time.time()
        if not isinstance(x,np.ndarray):  
            if self.hparam['decor']:
                xt = self.get_decor_factors(x)
            else:
                xt = self.get_factors(x)
        else:
            #print('aiiiiiiiiiiiiiiiiiiiiii')
            xt=x
        pred1 = self.meanPredictor.predict(xt, )  # self.reduce.get_factors(x))
        pred = type(self.reduce).inv_transform(self.reduce, pred1)
        if t: print('prediction time', self.name, time.time() - starttime)
        return maxi(0.01, pred)

    def variance(self, x):
        xt = self.get_var_factors(x)
        pred = self.predict(x)
        var = maxi(self.secondPredictor.predict(xt), 0.01)
        var = maxi(pred - pred ** 2, var)
        return var

    def zero_prob(self, x):
        xt = self.get_factors(x)
        p = self.predict(x)
        res=maxi(self.secondPredictor.predict(xt),np.exp(-p))
        return mini(maxi(res, 0.01), 0.99)

    def save(self):
        self.reduce.hours=self.hours
        self.meanPredictor.hours = self.hours
        # print('predictor dimension',self.meanPredictor.dim)
        type(self.reduce).save(self.reduce, add_path=self.env.system)
        self.meanPredictor.save(add_path=self.reduce.algo + self.env.system)
        if self.hparam['decor']:
            joblib.dump(self.featurePCA,config.root_path+'featurePCA')
            np.save(config.root_path+'norm_features_var',self.sum)
            np.save(config.root_path+'norm_features_mean',self.mean)
        if self.hparam['var'] or self.hparam['zero_prob']:
            self.secondPredictor.save(add_path=self.reduce.algo + self.meanPredictor.algo + self.env.system)

    def load(self):
        self.reduce.load(add_path=self.env.system)
        self.meanPredictor.load(add_path=self.reduce.algo + self.env.system)
        if self.hparam['decor']:
            self.featurePCA = joblib.load(config.root_path+'featurePCA')
            self.sum = np.load(config.root_path+'norm_features_var.npy')
            self.mean = np.load(config.root_path+'norm_features_mean.npy')
        if self.hparam['var']:
            self.secondPredictor.load(add_path=self.reduce.algo + self.meanPredictor.algo + self.env.system)
        if self.hparam['zero_prob']:
            self.secondPredictor.load(add_path=self.reduce.algo + self.meanPredictor.algo + self.env.system)
        try:
            self.hours = self.reduce.hours
        except AttributeError:
            try:
                self.hours = self.meanPredictor.hours
            except AttributeError:
                self.hours=[]

    def reset(self):
        self.reduce = red.get_reduction(self.reduce.algo)(self.env, dim=self.dim)
        self.meanPredictor = pr.get_prediction(self.meanPredictor.algo)(dim=self.reduce.dim, **self.hparam['pred'])
        self.secondPredictor = pr.get_prediction(self.secondPredictor.algo)(
            dim=len(Data(self.env).get_stations_col(None)), kwargs=self.hparam)

    def load_or_train(self, data, **kwargs):
        self.hparam.update(kwargs)
        try:
            self.load()
        except (IOError,EOFError):
            self.train(data, **self.hparam['pred'])
            self.save()

    def get_y(self, x, since=None):
        return self.reduce.get_y(x, since)

    def train_inv(self, data):
        if self.hparam['decor']:
            x=self.get_decor_factors(data)
        else:
            x=self.get_factors(data)
        self.reduce.train_inv(self.meanPredictor.predict(x), data.get_miniOD([])[data.get_stations_col()].as_matrix())