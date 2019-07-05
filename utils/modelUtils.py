import csv
import os

import numpy as np
import pandas as pd
# from keras.initializers import Initializer
from pandas import DataFrame, Series
from sklearn.cluster import KMeans

import code_v1.config as config

os.chdir(config.root_path)


def save(model, station=None, cluster=None, path=None):
    if path is None:
        if ((station is None) and (cluster is None)):
            path = 'model_reseau/model_' + model.type + '/model'
        elif (cluster is None):
            path = 'model_reseau/model_' + model.type + '/station/model_' + str(station)
        else:
            if not (os.path.exists("model_reseau/model_" + model.type)):
                os.mkdir("model_reseau/model_" + model.type)
            if not (os.path.exists("model_reseau/model_" + model.type + "/" + cluster)):
                os.mkdir("model_reseau/model_" + model.type + "/" + cluster)
            if station is None:
                path = 'model_reseau/model_' + model.type + '/' + cluster + '/model'
            else:
                path = 'model_reseau/model_' + model.type + '/' + cluster + '/model_' + str(station)
    model.save(path + ".h5")


def load(type, station=None, cluster=None, path=None):
    from keras.models import load_model
    if path is None:
        if ((station is None) and (cluster is None)):
            path = 'model_reseau/model_' + type + '/model'

        elif (cluster is None):
            path = 'model_reseau/model_' + type + '/station/model_' + str(station)
        else:
            if not (os.path.exists("model_reseau/model_" + type + "/" + cluster)):
                os.mkdir("model_reseau/model_" + type + "/" + cluster)
            if station is None:
                path = 'model_reseau/model_' + type + '/' + cluster + '/model'
            else:
                path = 'model_reseau/model_' + type + '/' + cluster + '/model_' + str(station)
    model = load_model(path + ".h5")
    model.type = type
    return model


def savecsv(data):
    if isinstance(data, DataFrame):
        data.to_csv(config.root_path + 'temp.csv', index=True)
    elif isinstance(data, Series):
        data.to_csv(config.root_path + 'temp.csv', index=True)
    elif isinstance(data, list):
        with open(config.root_path + 'temp.csv', 'wb') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(data)
    elif isinstance(data, np.ndarray):
        np.savetxt('temp.csv', data, delimiter=",")
    else:
        print('failed ' + str(type(data)))


def mape(y_real,y_pred,v=None,axis=None):
    return (np.abs(y_pred-y_real)/(y_real+1)).mean(axis=axis)

def mpe(y_real,y_pred,v=None, axis=None):
    return ((y_pred - y_real) / (y_real + 1)).mean(axis=axis)

def rmsle(y_real, y_pred, v=None,axis=None):
    # print(y_pred[:10], y_real[:10])
    y_pred = y_pred * (y_pred > 0)
    y_real = y_real * (y_real > 0)
    return \
        np.sqrt(
            np.mean(
                (
                    np.log((y_real) + 1) - np.log((y_pred) + 1)
                ) ** 2,
                axis=axis
            )
        )

def deviation(y_real,y_pred,v=None,axis=None):
    if isinstance(y_pred, DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_real, DataFrame):
        y_real = y_real.to_numpy()
    return (y_real - y_pred).mean(axis=axis)

def r_squared(y_real, y_pred,v=None, axis=None):
    if isinstance(y_pred, DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_real, DataFrame):
        y_real = y_real.to_numpy()
    y_bar = y_real.mean()
    SS_tot = ((y_real - y_bar) ** 2).sum(axis=axis)
    # SS_reg = ((y_pred - y_bar) ** 2).sum(axis=axis)
    SS_res = ((y_real - y_pred) ** 2).sum(axis=axis)
    return 1 - SS_res / SS_tot


def rmse(y_real, y_pred,v=None, axis=None):
    return np.sqrt(mse(y_real, y_pred, axis=axis))


def mse(y_real, y_pred, v=None,axis=None):
    return np.mean((y_real - y_pred) ** 2, axis=axis)


def rmse_per(y_real, y_pred,v=None, axis=None):
    return np.sqrt(np.mean((y_real - y_pred) ** 2/(y_real+1), axis=axis))# / (y_real + 1).mean(axis=axis)


def rmse_per_1(y_real, y_pred,v=None, axis=None):
    return (np.sqrt(np.mean((y_real - y_pred) ** 2, axis=axis)) / (y_real + 1).mean(axis=axis))[
        y_real.mean(axis=axis) < 1]


def rmse_per_2(y_real, y_pred,v=None, axis=None):
    return (np.sqrt(np.mean((y_real - y_pred) ** 2, axis=axis)) / (y_real + 1).mean(axis=axis))[
        (y_real.mean(axis=axis) < 2) & (y_real.mean(axis=axis) >= 1)]


def rmse_per_3(y_real, y_pred,v=None, axis=None):
    return (np.sqrt(np.mean((y_real - y_pred) ** 2, axis=axis)) / (y_real + 1).mean(axis=axis))[
        y_real.mean(axis=axis) >= 2]


def mae(y_real, y_pred, v=None, axis=None):
    return np.mean(np.abs(y_real - y_pred), axis=axis)


def transform(y, k):
    if isinstance(y, DataFrame):
        y = y.to_numpy()
    if (len(y.shape)) == 2:
        res = np.zeros((y.shape[0] - k + 1, y.shape[1]))
        for i in range(k):
            res += y[i:y.shape[0] - k + 1 + i, :]
        res = res / k
        return res
    else:
        res = np.zeros((y.shape[0] - k + 1))
        for i in range(k):
            res += y[i:y.shape[0] - k + 1 + i]
        res = res / k
        return res


def rmske(y_real, y_pred, k, axis=None):
    return rmse(transform(y_pred, k), transform(y_real, k), axis)


def make(y_real, y_pred, k, axis=None):
    return mae(transform(y_pred, k), transform(y_real, k), axis)


def rms5e(y_real, y_pred, axis=None):
    return rmske(y_real, y_pred, 5, axis)


def rms3e(y_real, y_pred, axis=None):
    return rmske(y_real, y_pred, 3, axis)


def ma3e(y_real, y_pred, axis=None):
    return make(y_real, y_pred, 3, axis)


def rmsGammaE(y_real, y_pred, axis=None, gamma=0.5):
    i = 1
    k = 0
    s = 1
    while i > 0.1:
        k += 1
        i *= gamma
        s += 2 * i
    del i

    def transform(y):
        if isinstance(y, DataFrame):
            y = y.to_numpy()
        if (len(y.shape)) == 2:
            res = np.zeros((y.shape[0] - (2 * k + 1) + 1, y.shape[1]))
            for i in range(2 * k + 1):
                res += gamma ** (abs(i - k)) * y[i:y.shape[0] - (2 * k + 1) + 1 + i, :]
            res = res / s
            return res
        else:
            res = np.zeros((y.shape[0] - (2 * k + 1) + 1))
            for i in range(2 * k + 1):
                res += gamma ** (abs(i - k)) * y[i:y.shape[0] - (2 * k + 1) + 1 + i]
            res = res / s
            return res

    return rmse(transform(y_pred), transform(y_real), axis)


def err_(y_real, y_pred):
    if isinstance(y_pred, DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_real, DataFrame):
        y_real = y_real.to_numpy()
    return np.array(rmsle(y_real, y_pred), rmse(y_real, y_pred), mae(y_real, y_pred), rmse_per(y_real, y_pred),
                    r_squared(y_real, y_pred))


class Normalizer(object):
    def __init__(self, x, axis=0):
        self.mean = x.mean(axis=axis)
        self.var = np.var(x, axis=axis)

    def transform(self, x):
        return (x - self.mean) / np.sqrt(self.var)


def compute_cost(OD, labels):
    s = 0
    for i in range(labels.shape[0]):
        for j in range(i + 1, labels.shape[0]):
            if labels[i] != labels[j]:
                s += OD[i, j] + OD[j, i]
    return s


def compute_relative_cost(OD, labels):
    c = compute_cost(OD, labels)
    return c / np.sum(OD)

def mini(a, b):
    inf = a < b
    return a * inf + b * (1 - inf)


def maxi(a, b):
    sup = a > b
    return a * sup + b * (1 - sup)


def normalize(array):
    array = mini(np.abs(array), 10)
    array = np.log(array + 1)
    return (array) / (np.max(array))


def laplace(a, moy=0, b=1):
    return 1 / (2 * b) * np.exp(-np.abs(a - moy) / b)


def dirac(a):
    return (a == 0) * 1


def png_to_gif(path):
    import imageio

    images = []
    os.listdir(path)
    for filename in os.listdir(path):
        if filename.__contains__('.png'):
            images.append(imageio.imread(path + filename))
    kwargs = {'fps': 2.0}
    imageio.mimsave(path + 'gif.gif', images, 'GIF-FI', **kwargs)


def sum_diag(mat):
    res = np.zeros(mat.shape[0] + mat.shape[1] - 1)

    for i in range(res.shape[0]):
        res[i] = (np.diag(mat, k=i - mat.shape[0] + 1)).sum()

    return res


def proba_sum(p1, p2):
    return sum_diag(np.dot(np.matrix(np.flipud(p1)).T, [p2]))


def proba_diff(p1, p2):
    return sum_diag(np.dot(np.matrix(p1).T, [p2]))


def to_one_dim(data):
    miniOD = data.get_miniOD(2015)
    st = data.get_stations_col(2015)
    a = np.zeros(
        (miniOD.shape[0] * len(data.get_stations_col(2015)), miniOD.shape[1] - len(data.get_stations_col(2015)) + 1))
    l = [x for x in miniOD.columns.values if not (st.__contains__(x))]
    l.append('End date')
    res = pd.DataFrame(a, columns=l)
    k = 0
    l = [x for x in miniOD.columns.values if not (st.__contains__(x))]
    for s in data.get_stations_col(2015):
        res.loc[k * miniOD.shape[0]:(k + 1) * miniOD.shape[0], l] = miniOD[l]
        res.loc[k * miniOD.shape[0]:(k + 1) * miniOD.shape[0], 'End date'] = miniOD[s]
    return res
