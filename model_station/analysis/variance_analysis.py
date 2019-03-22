from sklearn.decomposition import PCA

import model_station.Prediction as pr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import model_station.Prediction as pr
from utils.modelUtils import *
from model_station.ModelStations import ModelStations
from model_station.evaluate1model import EvaluateModel


def compute_errors(test_data, squared=True):
    mod = ModelStations(test_data.env, 'svd', 'gbt', dim=10)
    mod.load()
    pred = pd.DataFrame(mod.predict(test_data), columns=test_data.get_stations_col(2015))
    err = test_data.get_miniOD([])
    test_data.miniOD = None
    err[test_data.get_stations_col(2015)] -= pred
    if squared:
        err[test_data.get_stations_col(2015)] = err[test_data.get_stations_col(2015)] ** 2
    else:
        err[test_data.get_stations_col(2015)] = err[test_data.get_stations_col(2015)]
    return err

def plot_err(test_data):
    err = compute_errors(test_data, squared=False)
    col = ['Heure','Jour','Mois','Annee','precip','brr','Hum','temp','vent','visi','pression','vac']
    for ch in col:
        print(ch)
        sns.kdeplot(err[ch], err["Start date 10002"])
        plt.show()


def significativity(test_data, station=None):
    err = compute_errors(test_data)
    if station == None:
        data = test_data.get_ODsum()
        data['End date'] = err[test_data.get_stations_col(2015)].mean(axis=1)
        data.drop('Start date', axis=1, inplace=True)
    else:
        data = test_data.get_ODsum()
        data['End date'] = err[station].mean(axis=1)
        data.drop('Start date', axis=1, inplace=True)
    l = [x for x in data.columns.values if not (x.__contains__('date'))]
    for s in ['ch', 'precip', 'brr', 'LV', 'MMJ', 'SD', 'verglas', 'poudrerie', 'UTC timestamp']:
        try:
            l.remove(s)
        except ValueError:
            pass
    for h in range(24):
        l.remove('h' + str(h))
    pca = PCA()
    sum = data[l].as_matrix().sum(axis=0)
    p = pca.fit_transform(data[l].as_matrix() / sum)
    p = pd.DataFrame(p)
    p = data[l] / sum
    p['intercept'] = 1
    y = data['End date'].as_matrix().flatten()
    from statsmodels.regression.linear_model import OLS
    x = p
    x = x.astype(float)
    lin = OLS(y, x)
    res = lin.fit()
    print(res.summary())


def chose_model(test_data):
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)
    mod = ModelStations(train.env, 'svd', 'gbt', dim=10)
    mod.load_or_train(train)
    WH = mod.get_factors(test)
    res = mod.reduce.inv_transform(mod.meanPredictor.predict(WH))
    e = (maxi(res, 0.01) - mod.get_objectives(test)) ** 2
    d = {
        'linear': {},
        'decisiontree': {
            'min_samples_leaf': 140,
            'max_depth': 10,
        },
        'ridge': {'alpha': 360},
        'gbt': {
            'n_estimators': 10,
            'lr': 0.5,
            'max_depth': 1,
        },
        'randforest': {
            'n_estimators': 29,
            'min_samples_leaf': 20,
            'max_depth': 7,
        },
        # 'MLPvar': {''},
        'mean': {}
    }
    for a in ['linear', 'decisiontree', 'randforest', 'mean', 'gbt']:
        mod.secondPredictor = pr.get_prediction(a)(dim=len(train.get_stations_col(2015)), **d[a])
        mod.train_variance(train)
        var = mod.variance(test)
        print(a, rmse(e.as_matrix(), var))


def grid_search_dt(test_data):  # best sol depth 17, min_sample 80
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)
    mod = ModelStations(train.env, 'svd', 'gbt', dim=10)
    mod.load_or_train(train)
    WH = mod.get_factors(test)
    res = mod.reduce.inv_transform(mod.meanPredictor.predict(WH))
    e = (maxi(res, 0.01) - mod.get_objectives(test)) ** 2
    rmin = 20
    for m in range(20, 1000, 15):
        for depth in range(1, 20):
            d = {'min_samples_leaf': m,
                 'max_depth': depth,
                 }
            mod.secondPredictor = pr.get_prediction('decisiontree')(dim=len(train.get_stations_col(2015)), **d)
            mod.train_variance(train)
            var = mod.variance(test)
            r = rmse(e.as_matrix(), var)
            print((m, depth), r)
            if r < rmin:
                rmin = r
                amin = (depth, m)
    print(amin, rmin)


def estimateur_sans_biais(test_data):
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)
    mod = ModelStations(train.env, 'svd', 'gbt', dim=10)
    mod.load_or_train(train)
    WH = mod.get_factors(train)
    res = mod.predict(WH)
    e = (maxi(res, 0.01) - mod.get_objectives(train)) ** 2
    v = e.as_matrix().sum() / (e.shape[0] * e.shape[1] - 1)
    print(v)

    WH = mod.get_factors(test)
    res = mod.predict(WH)
    e = ((maxi(res, 0.01) - mod.get_objectives(test)) ** 2) - v
    print(rmse(v, e.as_matrix()))


def grid_search_ridge(test_data):  # best sol
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)
    mod = ModelStations(train.env, 'svd', 'gbt', dim=10)
    mod.load_or_train(train)
    WH = mod.get_factors(test)
    res = mod.reduce.inv_transform(mod.meanPredictor.predict(WH))
    e = ((maxi(res, 0.01) - mod.get_objectives(test)) ** 2).as_matrix()
    rmin = 20
    # for alpha in np.logspace(-10, 8, 50):
    for alpha in np.linspace(100, 500, 50):
        d = {'alpha': alpha,
             }
        mod.secondPredictor = pr.get_prediction('ridge')(dim=len(train.get_stations_col(2015)), **d)
        mod.train_variance(train)
        var = mod.variance(test)
        r = rmse(e, var)
        print((alpha), r)
        if r < rmin:
            rmin = r
            amin = (alpha)
    print(amin, rmin)


def grid_search_gbt(test_data):  # best sol (8, 0.1, 3)
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)
    mod = ModelStations(train.env, 'svd', 'gbt', dim=10)
    mod.train(train)
    WH = mod.get_factors(test)
    res = mod.reduce.inv_transform(mod.meanPredictor.predict(WH))
    e = (maxi(res, 0.01) - mod.get_objectives(test)) ** 2
    rmin = 20
    amin = None
    for n_estim in range(5, 30, 3):
        for lr in [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]:
            for md in range(1, 20, 2):
                hparam = {
                    'n_estimators': n_estim,
                    'lr': lr,
                    'subsample': 1,
                    'max_depth': md,
                }
                mod.secondPredictor = pr.get_prediction('gbt')(dim=len(train.get_stations_col(2015)), **hparam)
                mod.train_variance(train)
                var = mod.variance(test)
                r = rmse(e.as_matrix(), var)
                print((n_estim, lr, md), r, amin, rmin)
                if r < rmin:
                    rmin = r
                    amin = (n_estim, lr, md)
    print(amin, rmin)


def grid_search_rf(test_data):  # best sol (20, 35, 11)
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)
    mod = ModelStations(train.env, 'svd', 'gbt', dim=10)
    mod.load_or_train(train)
    WH = mod.get_factors(test)
    res = mod.reduce.inv_transform(mod.meanPredictor.predict(WH))
    e = (maxi(res, 0.01) - mod.get_objectives(test)) ** 2
    rmin = 20
    amin = None
    for md in range(3, 20, 2):
        for n_estim in range(5, 30, 3):
            for m in range(20, 300, 15):
                hparam = {
                    'n_estimators': n_estim,
                    'min_samples_leaf': m,
                    'max_depth': md,
                }
                mod.secondPredictor = pr.get_prediction('randforest')(dim=len(train.get_stations_col(2015)), **hparam)
                mod.train_variance(train)
                var = mod.variance(test)
                r = rmse(e.as_matrix(), var)
                print((n_estim, m, md), r, amin, rmin)
                if r < rmin:
                    rmin = r
                    amin = (n_estim, m, md)
    print(amin, rmin)


def ry_one(test_data, red, pred, var, hparam, red_dim=5, comb=0):
    train = test_data.get_partialdata_per(0, 0.8)
    test = test_data.get_partialdata_per(0.8, 1)

    # hparam['var']=True
    eM = EvaluateModel(train.env, red, pred, red_dim=red_dim, **{'var':True,'load_red': True, 'is_combined':comb})
    eM.mod.secondPredictor = pr.get_prediction(var)(dim=len(train.get_stations_col(None)),
                                                          **hparam)
    t = eM.mod.train(train, **{'verb': 2})
    # ti = time.time()
    p = eM.mod.predict(test)
    # tpred = time.time() - ti
    v = eM.mod.variance(test)
    e = (test.get_miniOD([],None)[test.get_stations_col(None)]-p)**2
    err = [rmsle(e, v).mean(), rmse(e, v).mean(), mae(e, v).mean(),
           mape(e, v).mean(), r_squared(e, v).mean()]
    del eM, test, train
    print(err)
    return err


def compute_scores():
    from preprocessing.Environment import Environment
    from preprocessing.Data import Data

    ud = Environment('Bixi', 'train')
    d = Data(ud)
    red_dim = {
        'svd': 5,
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
        'autoencoder',
        'svd',
        # 'pca',
        # 'id',
        # 'sum',
        'kmeans',
        # 'average',
        # 'complete',
        # 'weighted',
        # 'dep-arr',
        'GM',
        # 'SC',
        # 'maxkcut',
        # 'ODkmeans',
    ]
    pred = [
        # 'linear',
        'gbt',
        # 'ridge',
        # 'lasso',
        # 'MLP2',
        # 'MLP',
        # 'LSTM',
        # 'mean',
        'randforest',
        # 'decisiontree',
    ]
    var = [
        # 'linear',
        'gbt',
        # 'ridge',
        # 'lasso',
        # 'MLP2',
        # 'MLP',
        # 'LSTM',
        'mean',
        # 'randforest',
        'decisiontree',
    ]
    varparam = {
        'mean':{},
        'linear': {},
        'gbt': {
                    'n_estimators': 8,
                    'lr': 0.1,
                    'subsample': 1,
                    'max_depth': 3,
                },
        'randforest': {
                    'n_estimators': 30,
                    'min_samples_leaf': 20,
                    'max_depth': 7,
                },
        'decisiontree': {'min_samples_leaf': 140,
                 'max_depth': 10,
                 },
    }
    # f = open('errs_var.csv', 'w')
    # f.close()
    # for s in red:
    #     for p in pred:
    #         for v in var:
    #             with open('errs_var.csv', 'a') as f:
    #                 try:
    #                     r = red_dim[s]
    #                 except KeyError:
    #                     r = 10
    #                 res = ry_one(d, s, p, v,hparam=varparam[v], red_dim=r)
    #                 line = s + ',' + p + ','+v+','
    #                 for i in res:
    #                     line += str(i) + ','
    #                 line = line[:-1] + '\n'
    #                 f.write(line)
    for s in red:
        for p in pred:
    # s='svd'
    # p='randforest'
            for v in var:
                with open('errs_var.csv', 'a') as f:
                    # try:
                    #     r = red_dim[s]
                    # except KeyError:
                    #     r = 10
                    res = ry_one(d, s, p, v, hparam=varparam[v], red_dim=10)
                    line = s + ',' + p + ',' + v + ','
                    for i in res:
                        line += str(i) + ','
                    line = line[:-1] + '\n'
                    f.write(line)


def try_Poisson_global(test_data, red, pred, var, hparam, red_dim=5):
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)

    eM = EvaluateModel(train.env, red, pred, red_dim=red_dim, **{'load_red': True})
    eM.mod.varPredictor = pr.get_prediction('randforest')(dim=len(train.get_stations_col(2015)),)
    p = eM.mod.predict(test)
    v = eM.mod.variance(test)

if __name__ == '__main__':
    from preprocessing.Environment import Environment
    from preprocessing.Data import Data

    ud = Environment('Bixi', 'train')
    d = Data(ud)
    compute_scores()
    # print(significativity(d))
