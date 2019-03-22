import pickle

import keras.optimizers
import numpy as np
from pandas import DataFrame, Series
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree.tree import DecisionTreeRegressor

from utils.modelUtils import rmse, rms5e
import pandas as pd
from config import root_path
from statsmodels.tsa.arima_model import ARIMA

def loc(self, add_path=''):
    """
    fucntion to define the location where to save and name the prediction model
    :param self:
    :param add_path:
    :return:
    """
    directory = 'model_station/prediction_models/'
    p = root_path + directory + self.algo + str(self.dim) + add_path
    # print(p)
    return p

"""
All prediction classes, they override the Prediction class and its fucntions
"""
class Prediction(object):
    def __init__(self, horizon=1, dim=1, add_path='', *args, **kwargs):
        self.algo = ''
        self.dim = dim
        self.location = loc(self, add_path)
        return

    ## train the prediction model
    def train(self, x, y, **kwargs):
        raise Exception('train not Implemented')

    # predict the objectives
    def predict(self, x, y=None):
        raise Exception('predict not Implemented')

    # evaluate the model
    # x: pandas dataframe
    # y1: pandas df or numpy array
    def evaluate(self, x, y=None):
        if isinstance(y, DataFrame):
            y1 = y.as_matrix()
        elif isinstance(y, Series):
            y1 = y.as_matrix()
        pred = self.predict(x, y1)
        return [rmse(pred, y1), rmse(np.exp(pred) - 1, np.exp(y1) - 1)]

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()

    def load(self, add_path=''):
        raise Exception('predict not Implemented')

    def reset(self):
        raise Exception('predict not Implemented')


class MeanPredictor(Prediction):
    def __init__(self, dim=1, add_path='', horizon=-1, *args, **kwargs):
        super(MeanPredictor, self).__init__()
        self.algo = 'mean'
        self.dim = dim
        self.location = loc(self)

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        file = open(self.location, 'rb')
        l = pickle.load(file)
        file.close()
        self.mean = l.mean
        self.dim = l.dim

    def reset(self):
        del self.mean

    def predict(self, x, y=None):
        pred = np.zeros(shape=(self.dim, x.shape[0]))
        for d in range(self.dim):
            pred[d] = self.mean[d]
        return pred.transpose()

    def train(self, x, y, verb=0, **kwargs):
        self.mean = []
        if self.dim > 1:
            for d in range(self.dim):
                if isinstance(y, DataFrame):
                    y1 = y.as_matrix()[:, d]
                else:
                    y1 = y[:, d]
                # learn = range(int(y1.shape[0] * 1))  # 0.8))
                self.mean.append(y1.mean())
        else:
            self.mean.append(y.mean())

class MeanHourPredictor(Prediction):
    def __init__(self, dim=1, add_path='', horizon=-1, *args, **kwargs):
        super(MeanHourPredictor, self).__init__()
        self.algo = 'meanhour'
        self.dim = dim
        self.location = loc(self)

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        file = open(self.location, 'rb')
        l = pickle.load(file)
        file.close()
        self.mean = l.mean
        self.dim = l.dim

    def reset(self):
        del self.mean

    def predict(self, x, y=None):
        pred = np.zeros(shape=( x.shape[0],self.dim))
        m = np.array(self.mean)
        if 'Heure' in x.columns.values:
            pred = m[:,x['Heure'].astype(int)]
        else :
            for i in range(24):
                pred[:,x['h'+str(i)]] = m[:,i]
        return pred.T

    def train(self, x, y, verb=0, **kwargs):
        self.mean = []
        if self.dim > 1:
            for d in range(self.dim):
                if isinstance(y, DataFrame):
                    y1 = y.as_matrix()[:, d]
                else:
                    y1 = y[:, d]
                m = np.zeros(24)
                if 'Heure' in x.columns.values:
                    for i in range(24):
                        m[i]=y1[x['Heure']==i].mean()
                else:
                    for i in range(24):
                        m[x['h'+str(i)]==1,:]=y1[i,:].mean()
                # learn = range(int(y1.shape[0] * 1))  # 0.8))
                self.mean.append(m)
        else:
            m = np.zeros(24)
            if 'Heure' in x.columns.values:
                for i in range(24):
                    m[i] = y[x['Heure'] == i].mean()
            else:
                for i in range(24):
                    m[i] = y[x['h' + str(i)]].mean()
            # learn = range(int(y1.shape[0] * 1))  # 0.8))
            self.mean.append(m)


class ARIMAPredictor(Prediction):
    def __init__(self, dim=1, add_path='', horizon=-1, *args, **kwargs):
        super(ARIMAPredictor, self).__init__()
        self.algo = 'ARIMA'
        self.dim = dim
        self.location = loc(self, add_path)
        self.models=[]
        self.hparam={'order':(6,1,6)}
        self.hparam.update(kwargs)
        # self.lr = ARIMA(endog=n_jobs=-1)

    def predict(self, x, y=None):
        pred = np.zeros(shape=(self.dim, x.shape[0]))
        for d in range(self.dim):
            if isinstance(x,pd.DataFrame):
                x= x.as_matrix()
            pred[d] = self.models[d].predict(exog=x)
        return pred.transpose()

    def train(self, x, y, **kwargs):
        # verb= kwargs['verb'].
        import joblib
        # def f(d):
        #     return self.train_one(d, x, y)
        # print('pred_dim', self.dim)
        # self.models=joblib.Parallel(n_jobs=7)(joblib.delayed(ARIMA.train_one)(self.hparam,d,x,y) for d in range(self.dim))
        for d in range(self.dim):
            self.models.append(self.train_one(self.hparam,d,x,y))
        # for d in range(self.dim):
        #     self.train_one(d, x, y)
        print('\n')

    @staticmethod
    def train_one(hparam, d, x, y):
        print('\r', d, end='')
        # sys.stdout.flush()
        if isinstance(y, DataFrame):
            if len(y.shape) == 1:
                y1 = y.as_matrix()
            else:
                # print(obj.shape)
                y1 = y.as_matrix()[:, d]
        elif isinstance(y, Series):
            if len(y.shape) == 1:
                y1 = y.as_matrix()
            else:
                y1 = y[:, d].as_matrix()
        else:
            if len(y.shape) == 1:
                y1 = y
            else:
                y1 = y[:, d]
        mod = ARIMA(endog=y1, order=hparam['order'])# ,exog=x.as_matrix())
        ## MODEL 1  ##
        mod.fit()
        return mod

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        file = open(self.location, 'rb')
        l = pickle.load(file)
        file.close()
        self.models = l.models
        self.dim = l.dim

    def reset(self):
        del self.models


class LinearPredictor(Prediction):
    def __init__(self, dim=1, add_path='', horizon=-1, *args, **kwargs):
        super(LinearPredictor, self).__init__()
        self.algo = 'linear'
        self.dim = dim
        self.location = loc(self, add_path)
        self.lr = LinearRegression(n_jobs=-1)

    def predict(self, x, y=None):
        return self.lr.predict(x.as_matrix())

    def train(self, x, y, **kwargs):
        self.lr.fit(x.as_matrix(), y)
        # print(self.lr.get_params())

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        file = open(self.location, 'rb')
        l = pickle.load(file)
        file.close()
        self.lr = l.lr
        self.dim = l.dim

    def reset(self):
        del self.lr


class RidgePredictor(Prediction):
    def __init__(self, dim=1, add_path='', horizon=-1, *args, **kwargs):
        super(RidgePredictor, self).__init__()
        self.algo = 'ridge'
        self.dim = dim
        self.location = loc(self, add_path)
        hparam = {
            'alpha': 5
        }
        hparam.update(**kwargs)
        self.lr = Ridge(alpha=hparam['alpha'])

    def predict(self, x, y=None):
        return self.lr.predict(X=x.as_matrix())

    def train(self, x, y, **kwargs):
        self.lr.fit(x.as_matrix(), y)

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        file = open(self.location, 'rb')
        l = pickle.load(file)
        file.close()
        self.lr = l.lr
        self.dim = l.dim

    def reset(self):
        del self.lr


class LassoPredictor(Prediction):
    def __init__(self, dim=1, add_path='', horizon=-1, *args, **kwargs):
        super(LassoPredictor, self).__init__()
        self.algo = 'lasso'
        self.dim = dim
        self.location = loc(self, add_path)
        hparam = {
            'alpha': 5
        }
        hparam.update(**kwargs)
        self.lr = Lasso(alpha=hparam['alpha'])

    def predict(self, x, y=None):
        return self.lr.predict(x.as_matrix())

    def train(self, x, y, **kwargs):
        self.lr.fit(x.as_matrix(), y)

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        file = open(self.location, 'rb')
        l = pickle.load(file)
        file.close()
        self.lr = l.lr
        self.dim = l.dim

    def reset(self):
        del self.lr


class SvrLinear(Prediction):
    def __init__(self, dim=1, add_path='', horizon=-1, *args, **kwargs):
        super(SvrLinear, self).__init__()
        self.algo = 'svr_lin'
        self.dim = dim
        self.location = loc(self, add_path)
        self.svr = []
        hparam = {
            'kernel': 'linear',
            'C': 1,
            'max_iter': 300
        }
        hparam.update(**kwargs)
        for i in range(self.dim):
            self.svr.append(SVR(kernel=hparam['kernel'], C=hparam['C'], max_iter=hparam['max_iter']))

    def predict(self, x, y=None):
        pred = np.zeros(shape=(self.dim, x.shape[0]))
        for d in range(self.dim):
            pred[d] = self.svr[d].predict(x.as_matrix())
        return pred.transpose()

    def train(self, x, y, **kwargs):
        for d in range(self.dim):
            if isinstance(y, DataFrame):
                y1 = y[:, d].as_matrix()
            else:
                y1 = y[:, d]
            self.svr[d].fit(x.as_matrix(), y1)

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        file = open(self.location, 'rb')
        l = pickle.load(file)
        file.close()
        self.svr = l.svr
        self.dim = l.dim

    def reset(self):
        del self.svr


class SvrPoly(SvrLinear):
    def __init__(self, dim=1, add_path='', horizon=-1, *args, **kwargs):
        hparam = kwargs
        hparam['kernel'] = 'poly'
        super(SvrPoly, self).__init__(dim=1, add_path='', horizon=-1, *args, **kwargs)


class SvrRbf(SvrLinear):
    def __init__(self, dim=1, add_path='', horizon=-1, *args, **kwargs):
        hparam = kwargs
        hparam['kernel'] = 'rbf'
        super(SvrRbf, self).__init__(dim=1, add_path='', horizon=-1, *args, **kwargs)


class TripleGbt(Prediction):
    def __init__(self, horizon=1, dim=1, add_path='', *args, **kwargs):
        super(TripleGbt, self).__init__()
        self.horizon = horizon
        self.algo = '3GBT'
        self.dim = dim
        self.location = loc(self, add_path)
        self.model_GBT1 = []
        self.model_GBT2 = []
        self.model_GBT3 = []
        hparam = {
            'n_estimators': 0.2,
            'loss': 'ls',
            'subsample': 0.5,
            'learning_rate': 0.1,
        }
        hparam.update(**kwargs)

        # self.obj = objectives
        for i in range(self.dim):
            self.model_GBT1.append(
                GBR(n_estimators=hparam['n_estimators'], loss=hparam['loss'], subsample=hparam['subsample'],
                    learning_rate=hparam['learning_rate']))
            self.model_GBT2.append(
                GBR(n_estimators=hparam['n_estimators'], loss=hparam['loss'], subsample=hparam['subsample'],
                    learning_rate=hparam['learning_rate']))
            self.model_GBT3.append(
                GBR(n_estimators=hparam['n_estimators'], loss=hparam['loss'], subsample=hparam['subsample'],
                    learning_rate=hparam['learning_rate']))
        self.learning_days = [72, 49, 48, 22, 23, 24] + list(range(12))

    def predict(self, x, y=None):
        pred = np.zeros(shape=(self.dim, x.shape[0]))
        for d in range(self.dim):
            pred[d] = self.model_GBT1[d].predict(x.as_matrix())
            pred[d] += self.model_GBT2[d].predict(x.as_matrix())
            x['pred_GBT'] = pred[d]
            x['err'] = y[:, d] - pred[d]
            input3 = self.transform_data_2D(x)
            pred[d] += self.model_GBT3[d].predict(input3)
        return pred.transpose()

    def transform_data_2D(self, data):
        prev = self.learning_days
        x = np.zeros((data.shape[0], len(prev) * data.shape[1]))
        x[:5137] = self.transform_2d(data[:5137])
        x[5137:] = self.transform_2d(data[5137:])
        return x

    def transform_2d(self, data):
        prev = self.learning_days
        x = np.zeros((data.shape[0], (len(prev)) * data.shape[1]))
        l = len(prev)
        err, = np.where(data.columns.values == 'err')
        for k in range(l):
            a = data.shape[0] - prev[k]
            x[prev[k]:, k * data.shape[1]:(k + 1) * data.shape[1]] = data.as_matrix()[0:a, :]
            if prev[k] < self.horizon:
                x[:, k * data.shape[1] + err] = 0
        return x

    def train(self, x, y, verb=0, **kwargs):
        for d in range(self.dim):
            if isinstance(y, DataFrame):
                if len(y.shape) == 1:
                    y1 = y.as_matrix()
                else:
                    y1 = y[:, d].as_matrix()
            elif isinstance(y, Series):
                if len(y.shape) == 1:
                    y1 = y.as_matrix()
                else:
                    y1 = y[:, d].as_matrix()
            else:
                if len(y.shape) == 1:
                    y1 = y
                else:
                    y1 = y[:, d]
            learn = range(int(y1.shape[0]))

            # valid = range(int(y1.shape[0] * 0.8), int(y1.shape[0] * 0.9))

            # test = range(int(y1.shape[0] * 0.9), y1.shape[0])

            def print_perso(pred):
                # print('valid', rmse(pred[valid], y1[valid]))
                # print('valid5h', rms5e(pred[valid], y1[valid]))
                print('train', rmse(pred[learn], y1[learn]))
                print('train5h', rms5e(pred[learn], y1[learn]))

            ## MODEL 1  ##

            self.model_GBT1[d].fit(
                x.as_matrix()[learn],
                y1[learn]
                # np.log(y1[learn] + 1)
            )
            pred = self.model_GBT1[d].predict(x.as_matrix())
            # print_perso(pred)

            ##  MODEL 2  ##
            self.model_GBT2[d].fit(
                x.as_matrix()[learn],
                y1[learn] - pred[learn]
                # np.log(y1[learn] + 1) - pred[learn]
            )
            pred += self.model_GBT2[d].predict(x.as_matrix())
            # print_perso(pred)
            x.loc[:, 'pred_GBT'] = pred
            x.loc[:, 'err'] = y1 - pred
            # x['err'] = np.log(y1 + 1) - pred

            x_t = self.transform_data_2D(x)

            ##  MODEL 3  ##

            self.model_GBT3[d].fit(
                x_t[learn],
                y1[learn] - pred[learn]
                # np.log(y1[learn] + 1) - pred[learn]
            )
            pred += self.model_GBT3[d].predict(x_t)
            # print_perso(pred)

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        file = open(self.location, 'rb')
        l = pickle.load(file)
        file.close()
        self.model_GBT1 = l.model_GBT1
        self.model_GBT2 = l.model_GBT2
        self.model_GBT3 = l.model_GBT3
        self.dim = l.dim
        self.learning_days = l.learning_days

    def reset(self):
        del self.model_GBT1
        del self.model_GBT2
        del self.model_GBT3
        del self.learning_days


class DoubleGbt(Prediction):
    def __init__(self, horizon=1, dim=1, add_path='', *args, **kwargs):
        super(DoubleGbt, self).__init__()
        self.horizon = horizon
        self.algo = '2GBT'
        self.dim = dim
        self.location = loc(self, add_path)
        self.model_GBT1 = []
        self.model_GBT2 = []
        # self.obj = objectives
        hparam = {
            'n_estimators': 0.2,
            'loss': 'ls',
            'subsample': 0.5,
            'learning_rate': 0.1,
        }
        hparam.update(**kwargs)
        for i in range(self.dim):
            self.model_GBT1.append(
                GBR(n_estimators=hparam['n_estimators'], loss=hparam['loss'], subsample=hparam['subsample'],
                    learning_rate=hparam['learning_rate']))
            self.model_GBT2.append(
                GBR(n_estimators=hparam['n_estimators'], loss=hparam['loss'], subsample=hparam['subsample'],
                    learning_rate=hparam['learning_rate']))

    def predict(self, x, y=None):
        pred = np.zeros(shape=(self.dim, x.shape[0]))
        for d in range(self.dim):
            pred[d] = self.model_GBT1[d].predict(x.as_matrix())
            pred[d] += self.model_GBT2[d].predict(x.as_matrix())
        return pred.transpose()

    def train(self, x, y, verb=0, **kwargs):
        for d in range(self.dim):
            if isinstance(y, DataFrame):
                if len(y.shape) == 1:
                    y1 = y.as_matrix()
                else:
                    y1 = y[:, d].as_matrix()
            elif isinstance(y, Series):
                if len(y.shape) == 1:
                    y1 = y.as_matrix()
                else:
                    y1 = y[:, d].as_matrix()
            else:
                if len(y.shape) == 1:
                    y1 = y
                else:
                    y1 = y[:, d]
            learn = range(int(y1.shape[0] * 0.8))
            valid = range(int(y1.shape[0] * 0.8), int(y1.shape[0] * 0.9))

            # test = range(int(y1.shape[0] * 0.9), y1.shape[0])

            def print_perso(pred):
                print('valid', rms5e(pred[valid], y1[valid]))
                print('train', rms5e(pred[learn], y1[learn]))

            ## MODEL 1  ##

            self.model_GBT1[d].fit(
                x.as_matrix()[learn],
                y1[learn]
                # np.log(y1[learn] + 1)
            )
            pred = self.model_GBT1[d].predict(x.as_matrix())
            # print_perso(pred)

            ##  MODEL 2  ##
            self.model_GBT2[d].fit(
                x.as_matrix()[learn],
                y1[learn] - pred[learn]
                # np.log(y1[learn] + 1) - pred[learn]
            )
            pred += self.model_GBT2[d].predict(x.as_matrix())
            # print_perso(pred)

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        file = open(self.location, 'rb')
        l = pickle.load(file)
        file.close()
        self.model_GBT1 = l.model_GBT1
        self.model_GBT2 = l.model_GBT2
        self.dim = l.dim

    def reset(self):
        del self.model_GBT1
        del self.model_GBT2


class SimpleGbt(Prediction):
    def __init__(self, horizon=1, dim=1, add_path='', *args, **kwargs):
        super(SimpleGbt, self).__init__()
        self.horizon = horizon
        self.algo = 'gbt'
        self.dim = dim
        self.location = loc(self, add_path)
        self.model_GBT1 = []
        self.hparam = {
            'n_estimators': 150,
            'loss': 'ls',
            'subsample': 1,
            'lr': 0.1,
            'max_depth': 5
        }
        self.hparam.update(**kwargs)
        # for i in range(self.dim):
        #     self.model_GBT1.append(
        #         GBR(n_estimators=self.hparam['n_estimators'], loss=self.hparam['loss'], subsample=self.hparam['subsample'],
        #             learning_rate=self.hparam['lr'], max_depth=self.hparam['max_depth']))

    def predict(self, x, y=None):
        pred = np.zeros(shape=(self.dim, x.shape[0]))
        for d in range(self.dim):
            if isinstance(x,pd.DataFrame):
                x= x.as_matrix()
            pred[d] = self.model_GBT1[d].predict(x)
        return pred.transpose()

    def train(self, x, y, **kwargs):
        # verb= kwargs['verb'].
        import joblib
        # def f(d):
        #     return self.train_one(d, x, y)
        print('pred_dim', self.dim)
        self.model_GBT1=joblib.Parallel(n_jobs=min(self.dim,7))(joblib.delayed(SimpleGbt.train_one)(self.hparam,d,x,y) for d in range(self.dim))
        # for d in range(self.dim):
        #     self.train_one(d, x, y)
        print('\n')

    @staticmethod
    def train_one(hparam, d, x, y):
        print('\r', d, end='')
        # sys.stdout.flush()
        if isinstance(y, DataFrame):
            if len(y.shape) == 1:
                y1 = y.as_matrix()
            else:
                # print(obj.shape)
                y1 = y.as_matrix()[:, d]
        elif isinstance(y, Series):
            if len(y.shape) == 1:
                y1 = y.as_matrix()
            else:
                y1 = y[:, d].as_matrix()
        else:
            if len(y.shape) == 1:
                y1 = y
            else:
                y1 = y[:, d]
        gbt = GBR(n_estimators=hparam['n_estimators'], loss=hparam['loss'], subsample=hparam['subsample'],
                    learning_rate=hparam['lr'], max_depth=hparam['max_depth'])
        ## MODEL 1  ##
        gbt.fit(
            x.as_matrix(),
            y1
        )
        return gbt

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        file = open(self.location, 'rb')
        l = pickle.load(file)
        file.close()
        self.model_GBT1 = l.model_GBT1
        self.dim = l.dim

    def reset(self):
        del self.model_GBT1


class MLP(Prediction):
    def __init__(self, dim=10, horizon=-1, *args, **kwargs):
        super(MLP, self).__init__()
        self.algo = 'MLP'
        self.dim = dim
        self.model = None
        self.location = loc(self)
        self.hparam = kwargs

        ####################################
        ##           Main MODEL           ##
        ####################################

    def train(self, x, y, **kwargs):
        # from keras import losses
        from keras.callbacks import ReduceLROnPlateau
        from keras.layers import Input, Dense, Dropout, BatchNormalization
        from keras.layers.merge import Add
        from keras.models import Model
        from keras.utils import plot_model
        hparams = {
            'dropout': 0.5,
            'activation': 'relu',
            'noeuds': [60, 30, 15],
            'epsilon': 6.8129e-05,
            'learning_rate': 0.01,
            'reg_l1': 0,
            'reg_l2': 0.0000001,
            'beta_1': 0.977,
            'beta_2': 0.99973,
            'decay': 0.5,
            'n_couches': 3,
            'obj': 'mse',
            'epochs': 100,
            'batch_size': 32,
            'verb': 2
        }
        hparams.update(self.hparam)
        hparams.update(kwargs)
        for ch in x.columns.values:
            x[ch] = x[ch].astype('float64')
        x = x.as_matrix()
        lr = Ridge(alpha=0.2).fit(x, y)
        # validx = kwargs['validx']
        # validy = kwargs['validy']
        # print(validy.shape)
        dim = x.shape[1]
        n_out = self.dim
        input = Input(shape=(dim,))
        network = input
        for i in hparams['noeuds']:
            network = Dense(
                i,
                activation='relu',
                kernel_initializer=keras.initializers.normal(0, 0.5),
                bias_initializer='zeros',
            )(network)
            network = Dropout(hparams['dropout'])(network)
            network = BatchNormalization()(network)
        network = Dense(
            dim,
            activation='relu',
            kernel_initializer=keras.initializers.normal(0, 0.5),
            bias_initializer='zeros',
        )(network)
        network = Dropout(hparams['dropout'])(network)
        network = Add()([network, input])
        network = Dense(
            n_out,
            activation='linear',
            weights=(lr.coef_.T, lr.intercept_),
        )(network)
        self.model = Model(inputs=input, outputs=network)
        obj = hparams['obj']
        opt = keras.optimizers.Nadam(lr=hparams['learning_rate'])
        lrred = ReduceLROnPlateau(verbose=1, patience=20, min_lr=0.00000001)
        self.model.compile(optimizer=opt, loss=obj, metrics=['mae', 'mse'])
        plot_model(self.model, show_shapes=True, show_layer_names=True)
        self.model.fit(x, y,
                       epochs=hparams['epochs'],
                       batch_size=hparams['batch_size'],
                       verbose=hparams['verb'],
                       shuffle=True,
                       # validation_data=(validx.as_matrix(), validy),
                       callbacks=[lrred],
                       validation_split=0.2
                       )

    ####################################
    ##             Eval               ##
    ####################################
    def predict(self, x, y=None):
        if isinstance(x, DataFrame):
            x = x.as_matrix()
            x = x.astype('float64')
        # print(x.dtype)
        return self.model.predict(x)

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        # print(self.location)
        self.model.save(self.location + ".h5")

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        # print(self.location)
        self.model = keras.models.load_model(self.location + ".h5")

    def reset(self):
        del self.model


class MLP_var(MLP):
    def __init__(self, dim=10, horizon=-1, *args, **kwargs):
        super(MLP_var, self).__init__(dim=dim, horizon=horizon, *args, **kwargs)
        self.algo = 'MLPvar'
        self.dim = dim
        self.model = None
        self.location = loc(self)
        self.hparam = kwargs

        ####################################
        ##           Main MODEL           ##
        ####################################

    def train(self, x, y, **kwargs):
        # from keras import losses
        from keras.callbacks import ReduceLROnPlateau
        from keras.layers import Input, Dense, Dropout
        import keras.regularizers as regularizers
        from keras.models import Model
        from keras.utils import plot_model
        hparams = {
            'dropout': 0.99,
            'activation': 'relu',
            'noeuds': [20, 500],
            'epsilon': 6.8129e-05,
            'learning_rate': 0.01,
            'reg_l1': 0,
            'reg_l2': 0.0000001,
            'beta_1': 0.977,
            'beta_2': 0.99973,
            'decay': 0,
            'n_couches': 4,
            'obj': 'mse',
            'epochs': 100,
            'batch_size': 32,
            'verb': 2
        }
        hparams.update(self.hparam)
        hparams.update(kwargs)
        x.reset_index(inplace=True, drop=True)

        if isinstance(y, DataFrame):
            y.reset_index(inplace=True, drop=True)
            for ch in y.columns.values:
                y[ch] = y[ch].astype('float64')
            y = y.as_matrix()
        for ch in x.columns.values:
            x[ch] = x[ch].astype('float64')
        x = x.as_matrix()
        # lr = Ridge(alpha=0.2).fit(x, y)
        # validx = kwargs['validx']
        # validy = kwargs['validy']
        # print(validy.shape)
        dim = x.shape[1]
        n_out = self.dim
        import tensorflow as tf
        with tf.device("/cpu:0"):
            input = Input(shape=(dim,))
            network = input
            for i in hparams['noeuds']:
                network = Dense(
                    units=i,
                    activation='relu',
                    kernel_regularizer=regularizers.l1_l2(
                        hparams["reg_l1"],
                        hparams["reg_l2"])
                )(network)
                network = Dropout(hparams['dropout'])(network)
            # network = Add()([network, input])
            network = Dense(
                n_out,
                activation='linear',
                kernel_regularizer=regularizers.l1_l2(
                    hparams["reg_l1"],
                    hparams["reg_l2"])
            )(network)
            self.model = Model(inputs=input, outputs=network)
            obj = hparams['obj']
            opt = keras.optimizers.Nadam(lr=hparams['learning_rate'])
            lrred = ReduceLROnPlateau(verbose=1, patience=20, min_lr=0.00000001)
            self.model.compile(optimizer=opt, loss=obj, metrics=['mae', 'mse'])
            plot_model(self.model, show_shapes=True, show_layer_names=True)
            self.model.fit(x, y,
                           epochs=hparams['epochs'],
                           batch_size=hparams['batch_size'],
                           verbose=hparams['verb'],
                           shuffle=True,
                           # validation_data=(validx.as_matrix(), validy),
                           callbacks=[lrred],
                           validation_split=0.2
                           )

    ####################################
    ##             Eval               ##
    ####################################
    # def predict(self, x, y=None):
    #     if isinstance(x, DataFrame):
    #         x = x.as_matrix()
    #         x = x.astype('float64')
    #     # print(x.dtype)
    #     return self.model.predict(x)
    #
    # def save(self, add_path=''):
    #     self.location = loc(self, add_path)
    #     # print(self.location)
    #     self.model.save(self.location + ".h5")
    #
    # def load(self, add_path=''):
    #     self.location = loc(self, add_path)
    #     # print(self.location)
    #     self.model = keras.models.load_model(self.location + ".h5")
    #
    # def reset(self):
    #     del self.model

class MLP2(MLP):
    def __init__(self, dim=10, horizon=-1, **kwargs):
        super(MLP2, self).__init__()
        self.algo = 'MLP2'
        self.dim = dim
        self.model = None
        self.location = loc(self)
        self.hparams = {
            'verb': 2,
            'n_epochs': 200,
            'batch_size': 32,
            'dropout': 0.6,
            'activation': 'tanh',
            'noeuds': [60, 30, 15],
            'epsilon': 6.8129e-05,
            'learning_rate': 0.001,
            'reg_l1': 0,
            'reg_l2': 0.0000001,
            'beta_1': 0.977,
            'beta_2': 0.99973,
            'decay': 0.5,
            'n_couches': 3,
            'obj': 'mse'
        }
        self.hparams.update(**kwargs)

        ####################################
        ##           Main MODEL           ##
        ####################################

    def train(self, x, y, **kwargs):
        from keras import losses
        from keras.callbacks import ReduceLROnPlateau
        from keras.layers import Input, Dense, Dropout, BatchNormalization
        from keras.models import Model
        for ch in x.columns.values:
            x[ch] = x[ch].astype('float64')
        x = x.as_matrix()
        # validx = kwargs['validx'].as_matrix()
        # validy = kwargs['validy']
        dim = x.shape[1]
        n_out = self.dim
        # print(y1.shape, n_out)
        # print(x.shape, dim)
        # print(validx.shape)
        input = Input(shape=(dim,))
        init = keras.initializers.glorot_uniform(seed=None)
        network = input
        network = BatchNormalization()(network)
        for i in self.hparams['noeuds']:
            network = Dense(i,
                            activation=self.hparams['activation'],
                            kernel_regularizer=keras.regularizers.l1_l2(
                                self.hparams["reg_l1"],
                                self.hparams["reg_l2"]),
                            kernel_initializer=init,
                            )(network)
            network = Dropout(self.hparams['dropout'])(network)
            network = BatchNormalization()(network)
        network = Dense(n_out, activation='relu',
                        kernel_regularizer=keras.regularizers.l1_l2(
                            self.hparams["reg_l1"],
                            self.hparams["reg_l2"]),
                        kernel_initializer=init)(network)
        self.model = Model(inputs=input, outputs=network)

        # obj = losses.mean_squared_error
        # obj = losses.mean_squared_logarithmic_error

        opt = keras.optimizers.nadam(lr=self.hparams['learning_rate'],
                                     beta_1=self.hparams['beta_1'], beta_2=self.hparams['beta_2'],
                                     epsilon=self.hparams['epsilon'],
                                     schedule_decay=self.hparams['decay'])

        self.model.compile(
            optimizer=opt,
            loss=self.hparams['obj'],
            metrics=['mae', losses.mean_squared_logarithmic_error, 'mse'],
        )
        red = ReduceLROnPlateau()
        # print(x.shape)
        self.model.fit(x, y,
                       epochs=self.hparams['n_epochs'],
                       batch_size=self.hparams['batch_size'],
                       verbose=self.hparams['verb'],
                       shuffle=True,
                       validation_split=0.2,
                       callbacks=[red]
                       )

    ####################################
    ##             Eval               ##
    ####################################
    # def predict(self, x, y=None):
    #     if isinstance(x, DataFrame):
    #         x = x.as_matrix()
    #         x = x.astype('float64')
    #     # print(x.dtype)
    #     return self.model.predict(x)
    #
    # def save(self, add_path=''):
    #     self.location = loc(self, add_path)
    #     # print(self.location)
    #     self.model.save(self.location + ".h5")
    #
    # def load(self, add_path=''):
    #     self.location = loc(self, add_path)
    #     # print(self.location)
    #     self.model = keras.models.load_model(self.location + ".h5")
    #
    # def reset(self):
    #     del self.model


class MLP_kaggle(Prediction):
    def __init__(self, dim=10, horizon=-1, **kwargs):
        super(MLP_kaggle, self).__init__()
        self.algo = 'MLP_kaggle'
        self.dim = dim
        self.model = None
        self.location = loc(self)
        self.hparams = {
            'verb': 2,
            'n_epochs': 200,
            'batch_size': 32,
            'dropout': 0.8,
            'activation': 'relu',
            'noeuds': [500,250, 15],
            'epsilon': 6.8129e-05,
            'learning_rate': 0.011,
            'reg_l1': 0,
            'reg_l2': 0.0000001,
            'beta_1': 0.977,
            'beta_2': 0.99973,
            'decay': 0.,
            'n_couches': 3,
            'obj': 'mse'
        }
        self.hparams.update(**kwargs)

        ####################################
        ##           Main MODEL           ##
        ####################################

    def train(self, x, y, **kwargs):
        from keras import losses
        from keras.callbacks import ReduceLROnPlateau
        from keras.layers import Input, Dense, Dropout, BatchNormalization
        from keras.models import Model
        for ch in x.columns.values:
            x[ch] = x[ch].astype('float64')
        x = x.as_matrix()
        # validx = kwargs['validx'].as_matrix()
        # validy = kwargs['validy']
        dim = x.shape[1]
        n_out = self.dim
        # print(y1.shape, n_out)
        # print(x.shape, dim)
        # print(validx.shape)
        input = Input(shape=(dim,))
        init = keras.initializers.glorot_uniform(seed=None)
        network = input
        network = BatchNormalization()(network)
        for i in self.hparams['noeuds']:
            network = Dense(i,
                            activation=self.hparams['activation'],
                            kernel_regularizer=keras.regularizers.l1_l2(
                                self.hparams["reg_l1"],
                                self.hparams["reg_l2"]),
                            kernel_initializer=init,
                            )(network)
            network = Dropout(self.hparams['dropout'])(network)
            network = BatchNormalization()(network)
        network = Dense(n_out, activation='relu',
                        kernel_regularizer=keras.regularizers.l1_l2(
                            self.hparams["reg_l1"],
                            self.hparams["reg_l2"]),
                        kernel_initializer=init)(network)
        self.model = Model(inputs=input, outputs=network)

        # obj = losses.mean_squared_error
        # obj = losses.mean_squared_logarithmic_error

        # opt = keras.optimizers.nadam(lr=self.hparams['learning_rate'],
        #                              beta_1=self.hparams['beta_1'], beta_2=self.hparams['beta_2'],
        #                              epsilon=self.hparams['epsilon'],
        #                              schedule_decay=self.hparams['decay'])
        opt = keras.optimizers.adagrad(lr = self.hparams['learning_rate'],decay=self.hparams['decay'])
        self.model.compile(
            optimizer=opt,
            loss=self.hparams['obj'],
            metrics=['mae', losses.mean_squared_logarithmic_error, 'mse'],
        )
        red = ReduceLROnPlateau(patience=5)
        # print(x.shape)
        self.model.fit(x, y,
                       epochs=self.hparams['n_epochs'],
                       batch_size=self.hparams['batch_size'],
                       verbose=self.hparams['verb'],
                       shuffle=True,
                       validation_split=0.2,
                       callbacks=[red]
                       )

    ####################################
    ##             Eval               ##
    ####################################
    def predict(self, x, y=None):
        if isinstance(x, DataFrame):
            x = x.as_matrix()
            x = x.astype('float64')
        # print(x.dtype)
        return self.model.predict(x)

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        # print(self.location)
        self.model.save(self.location + ".h5")

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        # print(self.location)
        self.model = keras.models.load_model(self.location + ".h5")

    def reset(self):
        del self.model


class LSTMPredictor(Prediction):
    def __init__(self, dim=10, horizon=-1, **kwargs):
        super(LSTMPredictor, self).__init__()
        self.algo = 'LSTM24'
        self.dim = dim
        self.model = None
        self.location = loc(self)
        self.hparams = {
            'verb': 0,
            'window': 24,
            'n_epochs': 100,
            'batch_size': 32,
            'dropout': 0.6,
            'activation': 'relu',
            'noeuds': [60, 30, 15],
            'epsilon': 6.8129e-05,
            'learning_rate': 0.01,
            'reg_l1': 0,
            'reg_l2': 0.0000001,
            'beta_1': 0.977,
            'beta_2': 0.99973,
            'decay': 0.5,
            'n_couches': 3
        }
        self.hparams.update(kwargs)

    @staticmethod
    def transform_x(x, h):
        res = np.zeros((x.shape[0] - h + 1, h, x.shape[1]))
        for hour in range(h):
            res[:, hour, :] = x[hour:hour + res.shape[0], :]
        return res

    @staticmethod
    def inv_transform_x(x):
        h = x.shape[1]
        res = np.zeros((x.shape[0] + h - 1, x.shape[2]))
        res[h - 1:res.shape[0], :] = x[:, h - 1, :]
        res[0:h, :] = x[0, :, :]
        return res

    ####################################
    ##           Main MODEL           ##
    ####################################

    def train(self, x, y, **kwargs):
        from keras import losses
        from keras.layers import Input, Dense, LSTM, BatchNormalization
        from keras.models import Model

        for ch in x.columns.values:
            x[ch] = x[ch].astype('float64')
        x = x.as_matrix()
        x = self.transform_x(x, self.hparams['window'])
        y1 = self.transform_x(y, self.hparams['window'])
        validx = self.transform_x(kwargs['validx'].as_matrix(), self.hparams['window'])
        validy = self.transform_x(kwargs['validy'], self.hparams['window'])
        # validy = kwargs['validy']
        dim = x.shape[2]
        n_out = self.dim
        input = Input(shape=(self.hparams['window'], dim,))
        network = input
        network = BatchNormalization()(network)
        # init = keras.initializers.Orthogonal(gain=1.0, seed=None)
        init = keras.initializers.glorot_normal(seed=None)
        # init = keras.initializers.glorot_uniform(seed=None)
        # network = Dropout(0.1)(network)
        for i in range(self.hparams['n_couches']):
            network = LSTM(self.hparams['noeuds'][i],
                           activation=self.hparams['activation'],
                           kernel_regularizer=keras.regularizers.l1_l2(
                               self.hparams["reg_l1"],
                               self.hparams["reg_l2"]),
                           # kernel_initializer=init,
                           return_sequences=True,
                           )(network)
        network = BatchNormalization()(network)
        network = Dense(n_out, activation='relu',
                        kernel_regularizer=keras.regularizers.l1_l2(
                            self.hparams["reg_l1"],
                            self.hparams["reg_l2"]),
                        kernel_initializer=init)(network)
        self.model = Model(inputs=input, outputs=network)

        obj = losses.mean_squared_logarithmic_error

        opt = keras.optimizers.nadam(lr=self.hparams['learning_rate'],
                                     beta_1=self.hparams['beta_1'], beta_2=self.hparams['beta_2'],
                                     epsilon=self.hparams['epsilon'],
                                     schedule_decay=self.hparams['decay'])

        self.model.compile(optimizer=opt, loss=obj, metrics=['mae', 'mse'])
        # print(x.shape)
        # print(y1.shape)
        # print(validx.shape)
        # print(validy.shape)
        self.model.fit(x, y1,
                       epochs=self.hparams['n_epochs'],
                       batch_size=self.hparams['batch_size'],
                       verbose=self.hparams['verb'],
                       shuffle=True,
                       validation_data=(validx, validy)
                       )

    ####################################
    ##             Eval               ##
    ####################################
    def predict(self, x, y=None):
        if isinstance(x, DataFrame):
            x = x.as_matrix()
            x = x.astype('float64')
        # print(x.dtype)
        x = self.transform_x(x, self.hparams['window'])
        return self.inv_transform_x(self.model.predict(x))

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        # print(self.location)
        self.model.save(self.location + ".h5")

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        # print(self.location)
        self.model = keras.models.load_model(self.location + ".h5")

    def reset(self):
        del self.model


class RandForest(Prediction):
    def __init__(self, horizon=1, dim=1, add_path='', *args, **kwargs):
        super(RandForest, self).__init__()
        self.algo = 'randforest'
        self.dim = dim
        self.location = loc(self, add_path)
        hparam = {'min_samples_leaf': 3,
                  'max_depth': 14,
                  'n_estimators': 94
                  }
        hparam.update(**kwargs)
        self.RF = RF(n_estimators=hparam['n_estimators'], min_samples_leaf=hparam['min_samples_leaf'],
                     max_depth=hparam['max_depth'], n_jobs=-1)
        self.hparam = kwargs

    # train the prediction model
    def train(self, x, y, **kwargs):
        hparam = {}
        hparam.update(self.hparam)
        hparam.update(**kwargs)
        self.RF.fit(x.as_matrix(), y)

    # predict the objectives
    def predict(self, x, y=None):
        return self.RF.predict(x)

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        file = open(self.location, 'rb')
        l = pickle.load(file)
        file.close()
        self.RF = l.RF
        self.dim = l.dim

    def reset(self):
        del self.RF

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()


class DecisionTree(Prediction):
    def __init__(self, horizon=1, dim=1, add_path='', *args, **kwargs):
        super(DecisionTree, self).__init__()
        self.algo = 'decisiontree'
        self.dim = dim
        self.location = loc(self, add_path)
        hparam = {'min_samples_leaf': 11,
                  'max_depth': 20,
                  }
        hparam.update(**kwargs)
        self.DT = DecisionTreeRegressor(min_samples_leaf=hparam['min_samples_leaf'], max_depth=hparam['max_depth'])

    # train the prediction model
    def train(self, x, y, **kwargs):
        self.DT.fit(x.as_matrix(), y)

    # predict the objectives
    def predict(self, x, y=None):
        return self.DT.predict(x)

    def load(self, add_path=''):
        self.location = loc(self, add_path)
        file = open(self.location, 'rb')
        l = pickle.load(file)
        file.close()
        self.DT = l.DT
        self.dim = l.dim

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        model_file_save = open(self.location, 'wb')
        pickle.dump(self, model_file_save)
        model_file_save.close()

    def reset(self):
        del self.DT


algorithms = {
    'ARIMA':ARIMAPredictor,
    'linear':LinearPredictor,
    'ridge':RidgePredictor,
    'lasso':LassoPredictor,
    # 'svr_lin',
    # 'svr_poly',
    # 'svr_rbf',
    'MLPvar':MLP_var,
    # 'MLP2':MLP2,
    # 'LSTM':LSTMPredictor,
    # '3gbt':TripleGbt,
    # '2gbt':DoubleGbt,
    'mean':MeanPredictor,
    'meanhour':MeanHourPredictor,
    'randforest':RandForest,
    'decisiontree':DecisionTree,
    'gbt':SimpleGbt,
    '1gbt':SimpleGbt,
    'MLP':MLP,
}



def get_prediction(pred):
    if isinstance(pred, Prediction):
        return pred
    elif isinstance(pred, str):
        try:
            return algorithms[pred]
        except KeyError:
            raise Exception("could not interprete function " + pred)
    else:
        raise Exception("could not interprete function")
