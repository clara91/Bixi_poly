import time

import joblib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import sys
from utils.modelUtils import normalize
from config import root_path
from model_station.CombinedModelStation import CombinedModelStation
from model_station.evaluate1model import EvaluateModel
from preprocessing.Data import Data
from preprocessing.Environment import Environment

#print(sys.path)
sys.path.insert(0,'C:/Users/Clara Martins/Documents/Doutorado/Pierre Code/Bixi_poly/')


class EvaluatesModels(object):
    def __init__(self, training_data, pred_algos, red_algos, red_dim, comb, **kwargs):
        self.hparam = {
            'is_combined': 0,           #ensemble model param in [0,..,8]
            'load_red': False,          #load reduction method
            'is_model_station': True,   #local/global model
            'norm': False,              #normalize objectives
            'load': True,               #load model
            'hours': [],                #hours to consider as features
            'var': False,                #learn a variance predictor
            'zero_prob': False,          #learn a predictor that estimate the 0 probability
            'log': False,               #log transform the objectives
            'mean': False,              #
            'red': {},                  #reduction hparam
            'pred': {},                 #prediction hparam
            'second_pred': {            #variance/zero_prob hparam
                'pred': 'linear'
            }
        }
        self.hparam.update(kwargs)
        self.models = []
        if len(pred_algos) == 1:
            pred_algos = pred_algos * (len(red_algos))
        if len(red_algos) == 1:
            red_algos = red_algos * (len(pred_algos))
        self.algo = list(zip(red_algos, pred_algos))
        learned_red = []
        self.hparam['is_combined'] = False
        self.training_times = {}
        for red, pred in self.algo:
            if red in learned_red:
                self.hparam['load_red'] = True
            else:
                self.hparam['load_red'] = False
                learned_red.append(red)
            try:
                em = EvaluateModel(training_data.env, red, pred, red_dim[red], **self.hparam)
            except KeyError:
                em = EvaluateModel(training_data.env, red, pred, 10, **self.hparam)
            start = time.time()
            if self.hparam['load']:
                # pass
                em.mod.load_or_train(training_data)
                em.mod.reset()
            else:
                em.mod.train(data=training_data, **self.hparam)
                em.mod.save()
                em.mod.reset()
            self.training_times[red + ' ' + pred] = time.time() - start
            self.models.append(em)
        if comb:
            hparam['is_combined'] = comb
            em = EvaluateModel(training_data.env, None, None, red_dim=10, **self.hparam)
            self.models = [em] + self.models
            self.algo = [('c' + str(self.hparam['is_combined']), 'c' + str(self.hparam['is_combined']))] + self.algo

    def train(self, data, **kwargs):
        for mod in self.models:
            mod.train(data, **kwargs)

    def save(self):
        for mod in self.models:
            mod.save()

    def compute_err(self, test_data, err: list, station=None, axis=None):
        times = False
        if 'time' in err:
            times = True
            err.remove('time')
            s1 = self.models[0].compute_errors(test_data, err, station=station, axis=axis).shape
            s = (s1[0] + 2,)
            for i in range(1, len(s1)):
                s += (i,)
        else:
            s = self.models[0].compute_errors(test_data, err, station=station, axis=axis).shape
        s = (len(self.models),) + s
        error = np.zeros(s)
        for i, mod in enumerate(self.models):
            res = mod.compute_errors(test_data, err, station=station, axis=axis)
            if times:
                error[i, :-2] = res
                try:
                    error[i, -2] = self.training_times[mod.mod.name]
                except KeyError:
                    pass
                mod.mod.load()
                start = time.time()
                mod.mod.predict(test_data)
                error[i, -1] = time.time() - start
                mod.mod.reset()
            else:
                error[i] = res
        return error

    def compute_err_svdevol(self, test_data, err: list, max_week=10, station=None, axis=None):
        e =None
        k=0
        #print(self)
        test_data = test_data.get_partialdata_n((max_week-self.hparam['n_week'])*168,-1)
        for i in range(0,int(test_data.get_miniOD([]).shape[0]-self.hparam['n_week']*168),24):
        e = None
        k = 0
        test_data = test_data.get_partialdata_n(
            (max_week - self.hparam['n_week']) * 168, - 1)
        for i in range(0, int(test_data.get_miniOD([]).shape[0] - self.hparam['n_week'] * 168), 24):
            self.models[0].mod.load()
            train = test_data.get_partialdata_n(
                i, int(i + self.hparam['n_week'] * 168))
            # n=1-(168*self.hparam['n_week'])/data.get_miniOD([]).shape[0]
            #self.models[0].mod.train_inv(train)
            data=test_data.get_partialdata_n(int(i+self.hparam['n_week']*168),int(i+self.hparam['n_week']*168)+24)
            res= self.compute_err(data,err,station,axis)

            self.models[0].mod.train_inv(train)
            data = test_data.get_partialdata_n(
                int(i + self.hparam['n_week'] * 168), int(i + self.hparam['n_week'] * 168) + 24)
            res = self.compute_err(data, err, station, axis)

            k += 1
            print(k)
            if e is None:
                e = res
            else:
                e += res
        return e / k

    def compute_log_likelihood(self, test_data, distrib, station=None, axis=None):
        ll = []
        for mod in self.models:
            res = mod.log_likelyhood(test_data, distrib, station=station, axis=axis)
            ll.append(res)
        return ll

    def plot_station(self, train_data, test_data, station):
        for s in station:
            plt.figure(s, figsize=(10, 5))
            # print(self.compute_err(test_data, ['rmse','r2'], station=s))
            plt.clf()
            # plt.subplot(2,1,1)
            y = self.models[0].mod.get_y(test_data.get_miniOD(self.models[0].hparam['hours'])).to_numpy()
            plt.plot(y[300:600, s], label='real demand')
            plt.title(test_data.get_stations_col()[s])
            for mod in self.models:
                mod.plot(test_data, s)
            plt.legend()
            plt.xlabel('Time (hours)')
            plt.ylabel('Number of trips')
            # plt.subplot(2,1,2)
            # y = self.models[0].mod.get_y(train_data.get_miniOD(self.models[0].hparam['hours'])).to_numpy()
            # plt.plot(y[:, s], label='real demand')
            # plt.title(train_data.get_stations_col()[s])
            # for mod in self.models:
            #     mod.plot(train_data, s)
            # plt.legend()
            # plt.xlabel('Time (hours)')
            # plt.ylabel('Number of trips')
            plt.savefig(test_data.get_stations_col()[s].replace(' ', '_') + '.pdf', bbox_inches='tight')
            plt.show()

    def compute_residuals(self, test_data, station=None):
        residuals = []
        for mod in self.models:
            res = mod.residuals(test_data, station=station)
            residuals.append(res)
        return residuals

    def compute_normed_residuals(self, test_data, station=None):
        normed_residuals = []
        for mod in self.models:
            res = mod.normed_residuals(test_data, station=station)
            normed_residuals.append(res)
        return normed_residuals


def RMSE_vs_time(test_data, mods):
    residuals = mods.compute_residuals(test_data)
    w = 24
    for i in range(len(residuals)):
        res = (residuals[i][test_data.get_stations_col(2015)].to_numpy() ** 2).mean(axis=1)
        m = np.zeros(res.shape[0] - w)
        for h in range(w):
            m += res[h:-w + h]
        m /= w
        plt.plot(m, label=mods.names[i])
    plt.legend()
    plt.xlabel('time (hours)')
    plt.ylabel('MSE')
    plt.title('Error deviation (' + str(w) + ' hours rolling mean)')
    plt.show()


def err_vs_time(test_data, mods: EvaluatesModels, err=['rmse'], station=None):
    l = len(err)
    plt.figure(figsize=(10, l * 3))
    k = 0
    for e in err:
        k += 1
        plt.subplot(l, 1, k)
        scores = mods.compute_err(test_data, [e], station=station, axis=1)
        scores = scores.reshape((scores.shape[0], scores.shape[2]))
        w = 24
        for i in range(scores.shape[0]):
            res = scores[i]  # [test_data.get_stations_col(2015)].to_numpy() ** 2).mean(axis=1)
            m = np.zeros(res.shape[0] - w)
            for h in range(w):
                m += res[h:-w + h]
            m /= w
            plt.plot(m, label=mods.names[i])
        plt.legend()
        plt.xlabel('time (hours)')
        if e == 'llP':
            e = 'LogLikelihood Poisson'
        else:
            e = e.upper()
        plt.ylabel(e)
        plt.xticks(np.arange(0, scores.shape[1], step=168))
    plt.savefig('Error_deviation.pdf')
    plt.show()


def RMSE_norm_vs_time(test_data, mods):
    residuals = mods.compute_normed_residuals(test_data)
    w = 24
    for i in range(len(residuals)):
        res = (residuals[i][test_data.get_stations_col(2015)].to_numpy() ** 2).mean(axis=1)
        m = np.zeros(res.shape[0] - w)
        for h in range(w):
            m += res[h:-w + h]
        m /= w
        plt.plot(m, label=mods.names[i])
    plt.legend()
    plt.xlabel('time (hours)')
    plt.ylabel('RMSE_norm')
    plt.title('Error deviation (' + str(w) + ' hours rolling mean)')
    plt.show()


def plot_residuals(test_data, mods):
    import config
    mods.compute_residuals(test_data)
    mods.compute_normed_residuals(test_data)
    for i in range(len(mods.residuals)):
        residual = mods.residuals[i]
        normed_residual = mods.normed_residuals[i]
        m = normed_residual[test_data.get_stations_col(None)].to_numpy().flatten()
        sns.kdeplot(m[np.abs(m) < 5], bw=.1)
        plt.show()
        for ch in config.learning_var:
            print(ch)
            print(test_data.get_miniOD(None)[test_data.get_stations_col(None)].to_numpy().mean())
            print(residual[test_data.get_stations_col(None)].to_numpy().mean())
            plt.subplot(121)
            for s in test_data.get_stations_col(2015):
                plt.scatter(residual[ch], residual[s])
            plt.subplot(122)
            for s in test_data.get_stations_col(2015):
                plt.scatter(residual[ch], normed_residual[s])
            plt.show()


def build_combined_model_rmse(data: Data, mods: EvaluatesModels, pred_algos: list, red_algos: list, **hparam):
    """
    function to build a combined model based on the rmse score
    :param data:
    :param mods:
    :param pred_algos:
    :param red_algos:
    :param hparam:
    :return:
    """
    rmse = np.array(mods.compute_err(data, ['rmse'], axis=0))
    s = rmse.shape
    s = (s[0], s[2])
    rmse = rmse.reshape(s)

    for k in range(5):
        r = np.argmin(rmse, axis=0)
        rmse[r, range(rmse.shape[1])] = 10000
        joblib.dump(r, root_path + 'model_station/prediction_models/combined_model/best_' + str(
            k) + data.env.system)

    joblib.dump(red_algos,
                root_path + 'model_station/prediction_models/combined_model/red' + data.env.system)
    joblib.dump(pred_algos,
                root_path + 'model_station/prediction_models/combined_model/pred' + data.env.system)
    a = []
    for mod in mods.models:
        a.append(mod.mod.reduce.dim)
    joblib.dump(a, root_path + 'model_station/prediction_models/combined_model/dims' + data.env.system)
    c = CombinedModelStation(data.env, **hparam)
    c.train_variance(data)
    c.save()


def build_combined_model_distrib(training_data, mods, pred_algos, red_algos):
    """
    build combined model using poisson distrib
    :param training_data:
    :param mods:
    :param pred_algos:
    :param red_algos:
    :return:
    """
    loglikeP = np.array(mods.compute_log_likelihood_poisson(training_data))
    LL = loglikeP
    LL[LL == 0] = -np.inf
    r = np.argmax(LL, axis=0)

    joblib.dump(red_algos, root_path + 'model_station/prediction_models/combined_model/red')
    joblib.dump(pred_algos, root_path + 'model_station/prediction_models/combined_model/pred')
    joblib.dump(r, root_path + 'model_station/prediction_models/combined_model/best')
    a = []
    for mod in mods.models:
        a.append(mod.mod.reduce.dim)
    joblib.dump(a, root_path + 'model_station/prediction_models/combined_model/dims')


def analyse_loglike(test_data, mods):
    """
    compare the likelihood of several models on stations (Poisson, NB and ZI distribution hypotheses)
    :param test_data:
    :param mods:
    :return:
    """
    l1 = list(map(lambda x: x + ' NB', mods.names))
    l1.extend(list(map(lambda x: x + ' ZI', mods.names)))
    l1.extend(list(map(lambda x: x + ' P', mods.names)))
    loglikeNB = np.array(mods.compute_log_likelihood(test_data, 'NB'))
    loglikeZI = np.array(mods.compute_log_likelihood(test_data, 'ZI'))
    loglikeP = np.array(mods.compute_log_likelihood(test_data, 'P'))
    # loglikeG = np.array(mods.compute_log_likelihood_gaussian(test_data))
    # loglikegeo = np.array(mods.compute_log_likelihood_geom(test_data))
    LL = np.zeros((loglikeNB.shape[0] * 3, loglikeNB.shape[1]))
    LL[:loglikeNB.shape[0], :] = loglikeNB
    LL[loglikeNB.shape[0]:2 * loglikeNB.shape[0], :] = loglikeZI
    LL[2 * loglikeNB.shape[0]:3 * loglikeNB.shape[0], :] = loglikeP
    # LL[3 * loglikeNB.shape[0]:4 * loglikeNB.shape[0], :] = loglikeG
    # LL[4 * llzi.shape[0]:, :] = np.array(mods.loglikegeo)
    print('mean per model', list(zip(np.ma.masked_invalid(LL).sum(axis=1), map(lambda x: x.mod.name, mods.models))))
    print('mean per distrib')
    print(np.ma.masked_invalid(LL[:loglikeNB.shape[0], :]).mean())
    print(np.ma.masked_invalid(LL[loglikeNB.shape[0]:loglikeNB.shape[0] * 2, :]).mean())
    print(np.ma.masked_invalid(LL[loglikeNB.shape[0] * 2:loglikeNB.shape[0] * 3, :]).mean())
    # print(np.nanmean(LL[1-np.isinf(LL)], axis=1))
    # print(np.nanmean(LL[LL != np.inf],axis=1))
    LL[np.isnan(LL)] = 0
    LL[np.isinf(LL)] = 0
    LL[LL == 0] = -np.inf
    r = np.argmax(LL, axis=0)
    # LL /= mx
    print('mean_best', np.mean(np.ma.masked_invalid(LL[r, range(LL.shape[1])])))
    mx = np.max(LL, axis=0)
    LL = LL / mx
    means = test_data.get_miniOD(None)[test_data.get_stations_col(None)].mean(axis=0).to_numpy()
    # for i in np.unique(r):
    #     print(means[r == i].max())
    print('mean NB', means[r < loglikeNB.shape[0]].mean())
    print('mean ZI', means[(r < 2 * loglikeNB.shape[0]) * (r > loglikeNB.shape[0])].mean())
    print('mean poisson', means[(r < 3 * loglikeNB.shape[0]) * (r > 2 * loglikeNB.shape[0])].mean())
    # print('mean ga', means[(r < 4 * llzi.shape[0]) * (r > 3 * llzi.shape[0])].mean())
    # print('mean Gaussian', means[r > 3 * loglikeNB.shape[0]].mean())
    print('model name, mean trips per model, LL/maxLL, N inf')
    for i in range(LL.shape[0]):
        print(l1[i], means[r == i].mean(), np.mean(np.ma.masked_invalid(LL[i, :])), np.sum(np.isinf(LL[i, :])))
        print(np.ma.corrcoef(np.ma.masked_invalid(LL[i, :]), means[:LL.shape[1]])[1, 0])
    plt.hist(r, bins=np.arange(-0.5, 3 * len(mods.names) + 1, 1))

    # l1.extend(list(map(lambda x: x + ' geo', mods.names)))
    # l1.extend(list(map(lambda x: x + ' G', mods.names)))
    plt.xticks(range(len(l1)), l1, rotation='vertical')
    plt.show()

    for m in mods.loglike:
        print(m)
        print(m[np.logical_not(np.isinf(m))].mean())


def analyse_loglike_poisson(test_data, mods):
    LL = mods.compute_log_likelihood(test_data, 'P')
    r = np.argmax(np.array(LL), axis=0)
    means = test_data.get_miniOD(None)[test_data.get_stations_col(None)].mean(axis=0).to_numpy()
    for i in np.unique(r):
        print(means[r == i].max())
    plt.hist(r, bins=np.arange(-0.5, len(mods.names) + 1, 1))
    l1 = mods.names[:]
    plt.xticks(range(len(mods.names)), l1, rotation='vertical')
    plt.show()
    for m in LL:
        print(m)
        print(m[np.logical_not(np.isinf(m))].mean())


def analyse_loglike_zero_inf(test_data, mods):
    mods.compute_log_likelihood(test_data, 'ZI')
    r = np.argmax(np.array(mods.loglike), axis=0)
    means = test_data.get_miniOD(None)[test_data.get_stations_col(None)].mean(axis=0).to_numpy()
    for i in np.unique(r):
        print(means[r == i].max())
    plt.hist(r, bins=np.arange(-0.5, len(mods.names) + 1, 1))
    plt.xticks(range(len(mods.names)), mods.names, rotation='vertical')
    plt.show()

    for m in mods.loglike:
        print(m)
        print(m[np.logical_not(np.isinf(m))].mean())


def analyse_loglike_NB(test_data, mods):
    mods.compute_log_likelihood(test_data, 'NB')
    r = np.argmax(np.array(mods.loglikeNB), axis=0)
    means = test_data.get_miniOD(None)[test_data.get_stations_col(None)].mean(axis=0).to_numpy()
    for i in np.unique(r):
        print(means[r == i].max())
    plt.hist(r, bins=np.arange(-0.5, len(mods.names) + 1, 1))
    plt.xticks(range(len(mods.names)), mods.names, rotation='vertical')
    plt.show()

    for m in mods.loglikeNB:
        print(m)
        print(m[np.logical_not(np.isinf(m))].mean())


def compute_measures(test_data, mods, path='', station=None):
    measures = ['rmsle', 'rmse', 'mae', 'rmse_norm', 'r2',
                # 'llP',
                # 'llZI',
                'dev', 'mpe', 'mape',
                # 'time'
                ]
    if 'time' in measures:
        tt = True
    else : tt = False
    # mods = EvaluatesModels(training_data, pred_algos, red_algos, red_dim, hparam['is_combined'], **hparam)
    # err = mods.compute_err_svdevol(test_data, measures, station=station)
    err = mods.compute_err(test_data, measures, station=station)
    else:
        tt = False
    err = mods.compute_err_svdevol(test_data, measures, station=station)
    if tt:
        measures.append('train_time')
        measures.append('test_time')
    df = pd.DataFrame(err, columns=measures, index=mods.algo)
    df = df.round(decimals=3)
    print(df)
    print('measures' + path + '.csv')
    return df
    # df.to_csv('measures'+path+'.csv')


def compare_model(training_data, test_data, pred_algos, red_algos, red_dim, stations=[], **kwargs):
    st = training_data.get_stations_col()
    for i, s in enumerate(stations):
        if isinstance(s, str):
            stations[i] = st.index(s)
    hparam = {
        'is_model_station': True,
        'norm': False,
        'mean': False,
        'load': True,
        'decor': False,
        'hours': [],
        'log': False,
        'is_combined': 3, #?????? 
        'red': {},
        'pred': {}
    }
    hparam.update(kwargs)
    # p = pred_algos * len(red_algos)
    # s = [t for t in red_algos for _ in pred_algos]
    # pred_algos = p
    # red_algos = s

    percentile = 0.05 ### ???????????????????????????????
    # mods = EvaluatesModels(training_data, pred_algos, red_algos, red_dim, False, **hparam)
    # mods.names = [str(mods.algo[i][0]) + ' ' + str(mods.algo[i][1]) for i in range(len(mods.algo))]
    # build_combined_model_rmse(training_data.get_partialdata_per(0.3,0.6), mods, pred_algos, red_algos)
    mods = EvaluatesModels(training_data, pred_algos, red_algos, red_dim, hparam['is_combined'], **hparam)
    # mods.models[0].train(data_train)
    # mods.models[0].mod.save()
    print(mods.algo)
    mods.names = [str(mods.algo[i][0]) + ' ' + str(mods.algo[i][1]) for i in range(len(mods.algo))]
    colors = [cm.jet(i) for i in np.linspace(0, 1, max(5, max(len(pred_algos), len(red_algos))))]
    # plot_residuals(test_data, mods)
    # analyse_loglike_zero_inf(test_data, mods)
    # analyse_loglike_NB(training_data, mods)
    # analyse_loglike_gaussian(training_data, mods)
    # err_vs_time(test_data,mods,['mape','llP','rmse','rmsle','r2'])
    # stations = [test_data.get_stations_col().index(i) if isinstance(i, str) else i for i in stations]
    # mods.plot_station(training_data,test_data, stations)
    df = None
    if not stations:

        return compute_measures(test_data, mods, path=str(hparam['log']) + str(hparam['decor']) + str(
            hparam['mean']) + training_data.env.system, station=[])
    m = test_data.get_miniOD([])[test_data.get_stations_col()].mean().to_numpy()
    for station in stations:
        if df is None:
            df = compute_measures(test_data, mods, path=str(hparam['log']) + str(hparam['decor']) + str(
                hparam['mean']) + training_data.env.system, station=station)
            df['station'] = test_data.get_stations_col()[station]
            df['size'] = m[station]
            df['algo'] = mods.names
        else:
            d = compute_measures(test_data, mods, path=str(hparam['log']) + str(hparam['decor']) + str(
                hparam['mean']) + training_data.env.system, station=station)
            d['station'] = station
            d['algo'] = mods.names
            d['station'] = test_data.get_stations_col()[station]
            d['size'] = m[station]
            df = df.append(d, ignore_index=True)
    return df
    # analyse_loglike_poisson(test_data, mods)
    # analyse_loglike(test_data, mods)
    # r_squared_analysis(test_data, mods)
    # r_squared_analysis(training_data, mods)

    # plt.title('mae')
    # for i in range(len(mods.rmse)):
    #     print(str(mods.algo[i][0]) + ' ' + str(mods.algo[i][1]))
    #     print('mae', mods.mae[i].to_numpy().mean())
    #     print('mae_3h', mods.mae_3h[i].mean())
    #     print('rmse', mods.rmse[i].to_numpy().mean())
    #     print('rmse 3h', mods.rmse_3h[i].mean())
    #     print('rmse_var', mods.rmse_var[i].to_numpy().mean())
    #     sns.kdeplot(mods.mae[i], c='r', label='mae')
    #     sns.kdeplot(mods.mae_3h[i], c='b', label='mae_3h')
    #     sns.kdeplot(mods.rmse[i], c='g', label='rmse')
    #     sns.kdeplot(mods.rmse_3h[i], c='k', label='rmse_3h')
    # plt.legend()
    # plt.show()


def redution_analysis(training_data, test_data, pred_algo, red_dim):
    import model_station.Reduction as s
    compare_model(training_data, test_data, [pred_algo] * (len(s.algorithms.keys())), s.algorithms.keys(), red_dim)


def prediction_analysis(training_data, test_data, red_algo, red_dim):
    import model_station.Prediction as s
    compare_model(training_data, test_data, s.algorithms.keys(), [red_algo] * (len(s.algorithms.keys())), red_dim)


def complete_analysis(training_data, test_data, red_dim):
    import model_station.Reduction as s
    import model_station.Prediction as p
    compare_model(training_data, test_data, p.algorithms.keys(), s.algorithms.keys(), red_dim)


def plot_one_day(data_train, data_test, pred_algo, red_algo, is_model_station=True, load=True, **kwargs):
    data = data_test.get_miniOD()
    # if (len(pred_algos) == 1):
    #     pred_algos = pred_algos * (len(red_algos))
    # if (len(red_algos) == 1):
    #     red_algos = red_algos * (len(pred_algos))
    year, month, day = 2017, 7, 1
    data = data[data['Annee'] == year]
    data = data[data['Mois'] == month]
    data = data[data['Jour'] == day]
    # print(data.shape)
    mod = EvaluateModel(data_train.env, red_algo, pred_algo, is_model_station, **kwargs)
    if load:
        mod.load(add_path=mod.env.system)
    else:
        mod.train(data_train)
        mod.save()
    p = pd.DataFrame(mod.predict(data), columns=data_train.get_stations_col(), index=data.index)
    diff = data[data_train.get_stations_col()] - p[data_train.get_stations_col()]
    sns.kdeplot(diff.to_numpy().flatten())
    plt.show()
    # print(diff.to_numpy().mean())
    diff_dep = normalize(diff[data_train.get_dep_cols(None)].to_numpy())
    diff_arr = normalize(diff[data_train.get_arr_cols(None)].to_numpy())

    loc = data_train.get_stations_loc()
    cmap = cm.jet
    for h in range(24):
        plt.figure(2 * h)
        plt.title('UTC timestamp' + str(h))
        plt.scatter(loc['lng'], loc['lat'], s=10,
                    c=diff_dep[h, :], cmap=cmap, alpha=0.5)
        plt.colorbar()
        if h < 10:
            plt.savefig('resultats/qualite_prediction/dep0' + str(h) + 'h.png')
        else:
            plt.savefig('resultats/qualite_prediction/dep' + str(h) + 'h.png')
        plt.figure(2 * h + 1)
        plt.title('UTC timestamp' + str(h))
        plt.scatter(loc['lng'], loc['lat'], s=10,
                    c=diff_arr[h, :], cmap=cmap, alpha=0.5)
        plt.colorbar()
        if h < 10:
            plt.savefig('resultats/qualite_prediction/arr0' + str(h) + 'h.png')
        else:
            plt.savefig('resultats/qualite_prediction/arr' + str(h) + 'h.png')
    plt.show()


def residual_plots(test_data, mods, station=None, squared=False):
    """
    displays the dependency between trip numbers and properties  
    :param learning_var: field of the analysis
    :param station: the station of the analysis, if none analysis performed on the total number of trips
    :return: none
    """
    import config
    learning_var = config.learning_var
    data = test_data.get_miniOD()
    mods.models[0].mod.load()
    pred = mods.models[0].mod.predict(test_data)
    test_data.miniOD = None
    if squared:
        data[test_data.get_stations_col(2015)] = (data[test_data.get_stations_col(2015)] - pred) ** 2  # /pred
    else:
        data[test_data.get_stations_col(2015)] = data[test_data.get_stations_col(2015)] - pred
    ind = data[data['Annee'] == 0].index
    data.drop(ind, inplace=True)
    print(data.columns.values)
    i = 0
    if station is None:
        ch_an = test_data.get_stations_col(2015)
    else:
        ch_an = 'End date ' + str(station)
    for ch in learning_var:
        if not (ch[0] == 'h') and not (ch in ['LV', 'MMJ', 'SD', 'poudrerie', 'verglas']):
            # data.boxplot(ch_an, by=ch)
            # if ch != 'Heure':
            plt.figure(i // 9)
            # plt.title('squared error / expectation')
            fig = plt.subplot(3, 3, (i % 9) + 1)
            i += 1
            # fig = plt.figure().add_subplot(111)
            fig.set_xlabel(ch)
            if squared:
                fig.set_ylabel('error²')
            else:
                fig.set_ylabel('error')
            l = []
            xaxis = np.unique(data[ch])
            print(ch, xaxis.shape)
            if xaxis.shape[0] < 20 or ch == 'Heure':
                for u in xaxis:
                    l.append(data[ch_an][data[ch] == u])
            else:
                m = np.min(data[ch])
                M = np.max(data[ch])
                step = (M - m) / 20
                xaxis = np.arange(m, M, step)
                for u in xaxis:
                    l.append(data[ch_an][(data[ch] >= u) * (data[ch] < u + step)])
            xaxis = xaxis.astype(int)
            # fig = plt.boxplot(ch_an, by=ch)
            # g = data.groupby(ch).mean()[ch_an]
            # v = data.groupby(ch).std()[ch_an]
            plt.boxplot(l, labels=xaxis)
            if squared:
                plt.ylim((0, 12))
            else:
                plt.ylim((-5, 5))
                # plt.plot(g, '-r')
                # plt.plot(g + v, ':r')
                # plt.plot(g - v, ':r')
    plt.show()


if __name__ == '__main__':
    data_train = Data(Environment('Bixi', 'train')).get_partialdata_per(0, 0.8)
    # print(data_train.miniOD.shape)
    data_test = Data(Environment('Bixi', 'train')).get_partialdata_per(0.8, 1) 
    # print(data_test.miniOD.shape)
    red_dim = {
        'svd': 10,
        'pca': 5,
        'autoencoder': 5,
        'kpca': 5,
        'kmeans': 10,
        'average': 10,
        'complete': 10,
        'weighted': 10,
        'GM': 10,
        'SC': 10,
        'maxkcut': 10,
        'ODkmeans': 10,
    }

    red = [
        # 'autoencoder',
        'svd',
        # 'pca',
        # 'id',
        # 'sum',
        'kmeans',
        # 'average',
        # 'complete',
        # 'weighted',
        # 'dep-arr',
        # 'GM',
        # 'SC',
        # 'maxkcut',
        # 'ODkmeans',
    ]
    pred = [
        # 'ARIMA',
        # 'linear',
        'gbt',
        # 'ridge',
        # 'lasso',
        # 'svr_lin',
        # 'svr_poly',
        # 'svr_rbf',
        # 'MLP2',
        # 'MLP',
        # 'LSTM',
        # '3gbt',
        # '2gbt',
        # 'gbt',
        # 'mean',
        'randforest',
        # 'decisiontree',
    ]

    hparam = {
        'is_model_station': True,
        'log': False,
        'mean': False,
        'load': False,
        'obj': 'mse',
        'hours': [],
        'decor': False,
        'n_week':4,
        'n_week': 4,
        'is_combined': 0,
    }
    with tf.device('/cpu:0'):
        # compare_model(data_train, data_test, pred, red, red_dim, stations=['Start date 6221', 'End date 6307', 'End date 5005'], **hparam)
        compare_model(data_train, data_test, pred, red, red_dim, stations=[], **hparam)
        # complete_analysis(data_train, data_test, load=False, norm=False)
        # complete_analysis(data_train, data_test, red_dim)
        # plot_one_day(data_train, data_test, pred[0], red[0], load=True)
