import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model_station.Prediction as pr
from model_station.evaluate1model import EvaluateModel
from random import shuffle

def grid_search_gbt(test_data, red):  # best sol (135, 0.1, 5)
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)
    eM = EvaluateModel(train.env, red, 'gbt', red_dim=5, **{'load_red': True})
    eM.mod.reduce.load(test_data.env.system)
    rmse_min = 999999999
    bestmod = None
    best_err = None
    for n_estim in range(5, 200, 10):
        for lr in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]:
            for md in range(1, 20):
                hparam = {
                    'n_estimators': n_estim,
                    'lr': lr,
                    'subsample': 1,
                    'max_depth': md,
                }
                eM.mod.meanPredictor = pr.get_prediction('gbt')(dim=eM.mod.reduce.dim, **hparam)
                # eM.pred = None
                eM.train(train)
                err = [eM.rmsle(test, False).mean(), eM.rmse(test, False).mean(), eM.mae(test, False).mean(),
                       eM.rmse_per(test, False).mean(), eM.r_squared(test, False).mean()]
                print((n_estim, lr, md), err, bestmod, best_err)
                if rmse_min > err[1]:
                    rmse_min = err[1]
                    bestmod = (n_estim, lr, md)
                    best_err = err
    print(bestmod, best_err)


def grid_search_ridge(test_data, red):  # best sol 2.55 (<150)
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)
    eM = EvaluateModel(train.env, red, 'ridge', red_dim=5, **{'load_red': True})
    eM.mod.reduce.load(test_data.env.system)
    rmse_min = 999999999
    bestmod = None
    best_err = None
    for alpha in np.logspace(-10, 20):
        hparam = {
            'alpha': alpha,
        }
        eM.mod.meanPredictor = pr.get_prediction('ridge')(dim=eM.mod.reduce.dim, **hparam)
        eM.train(train)
        err = [eM.rmsle(test, False).mean(), eM.rmse(test, False).mean(), eM.mae(test, False).mean(),
               eM.rmse_per(test, False).mean(), eM.r_squared(test, False).mean()]
        print((alpha), err, bestmod, best_err)
        if rmse_min > err[1]:
            rmse_min = err[1]
            bestmod = (alpha)
            best_err = err
    print(bestmod, best_err)


def grid_search_dt(test_data, red):  # best sol (11, 20)
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)
    eM = EvaluateModel(train.env, red, 'decisiontree', red_dim=5, **{'load_red': True})
    eM.mod.reduce.load(test_data.env.system)
    rmse_min = 999999999
    for depth in range(1, 20):
        for m in range(20, 1000, 15):
            hparam = {'min_samples_leaf': m,
                      'max_depth': depth,
                      }
            eM.mod.meanPredictor = pr.get_prediction('decisiontree')(dim=eM.mod.reduce.dim, **hparam)
            eM.train(train)
            # eM.pred = None
            err = [eM.rmsle(test, False).mean(), eM.rmse(test, False).mean(), eM.mae(test, False).mean(),
                   eM.rmse_per(test, False).mean(), eM.r_squared(test, False).mean()]
            print((depth, m), err)
            if rmse_min > err[1]:
                rmse_min = err[1]
                bestmod = (depth, m)
                best_err = err
    print(bestmod, best_err)


def grid_search_rf(test_data):  # best sol (3, 14, 92)
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)

    eM_svd = EvaluateModel(train.env, 'svd', 'randforest', red_dim=10, **{'load_red': True})
    eM_svd.train(train)
    # eM_svd.mod.reduce.load(test_data.env.system)
    eM_kmeans = EvaluateModel(train.env, 'kmeans', 'randforest', red_dim=10, **{'load_red': True})
    eM_kmeans.train(train)
    eM_auto = EvaluateModel(train.env, 'autoencoder', 'randforest', red_dim=10, **{'load_red': True})
    eM_auto.train(train)
    eM_GM = EvaluateModel(train.env, 'GM', 'randforest', red_dim=10, **{'load_red': True})
    eM_GM.train(train)

    # eM_kmeans.mod.reduce.load(test_data.env.system)
    rmse_min = 999999999
    bestmod = None
    best_err = None
    df = pd.DataFrame(columns=['max_depth','n_estimators','min_samples_leaf','kmeans','svd','GM','autoencoder'])
    li = [(d,m,l) for d in range(2, 20, 2) for m in np.logspace(0.2, 3, 20) for l in np.logspace(0.2, 5, 15)]
    shuffle(li)
    for (depth,m,leaf) in li:
        hparam = {
            'min_samples_leaf': int(leaf),
            'max_depth': depth,
            'n_estimators': int(m)
        }
        eM_svd.mod.meanPredictor = pr.get_prediction('randforest')(dim=eM_svd.mod.reduce.dim, **hparam)
        eM_svd.train(train)
        eM_kmeans.mod.meanPredictor = pr.get_prediction('randforest')(dim=eM_kmeans.mod.reduce.dim, **hparam)
        eM_kmeans.train(train)
        eM_GM.mod.meanPredictor = pr.get_prediction('randforest')(dim=eM_GM.mod.reduce.dim, **hparam)
        eM_GM.train(train)
        eM_auto.mod.meanPredictor = pr.get_prediction('randforest')(dim=eM_auto.mod.reduce.dim, **hparam)
        eM_auto.train(train)
        # eM.pred=None
        df.loc[df.shape[0]] = [depth, m, leaf, eM_kmeans.compute_errors(test, ['rmse'], load=False)[0],
                               eM_svd.compute_errors(test, ['rmse'], load=False)[0], eM_GM.compute_errors(test, ['rmse'], load=False)[0],
                               eM_auto.compute_errors(test, ['rmse'], load=False)[0]]
        # err = [eM_svd.rmsle(test, False).mean(), eM_svd.rmse(test, False).mean(), eM_svd.mae(test, False).mean(),
        #        eM_svd.rmse_per(test, False).mean(), eM_svd.r_squared(test, False).mean()]
        print((depth, int(m), int(leaf)))
        df.to_csv('svd_vs_kmeans')

    # for depth in range(2, 20, 2):
    #     for m in np.logspace(0.2, 3, 20):
    #         for leaf in np.logspace(0.2, 5, 15):
    #             hparam = {
    #                 'min_samples_leaf': int(leaf),
    #                 'max_depth': depth,
    #                 'n_estimators': int(m)
    #             }
    #             eM_svd.mod.meanPredictor = pr.get_prediction('randforest')(dim=eM_svd.mod.reduce.dim, **hparam)
    #             # print(1)
    #             eM_svd.train(train)
    #             eM_kmeans.mod.meanPredictor = pr.get_prediction('randforest')(dim=eM_kmeans.mod.reduce.dim, **hparam)
    #             # print(1)
    #             eM_kmeans.train(train)
    #             # eM.pred=None
    #             df.loc[df.shape[0]]=[depth,m,leaf,eM_kmeans.compute_errors(test,['rmse'],load=False),eM_svd.compute_errors(test,['rmse'],load=False)]
    #             # err = [eM_svd.rmsle(test, False).mean(), eM_svd.rmse(test, False).mean(), eM_svd.mae(test, False).mean(),
    #             #        eM_svd.rmse_per(test, False).mean(), eM_svd.r_squared(test, False).mean()]
    #             print((depth,int(m),int(leaf)))
    #             # if rmse_min > err[1]:
    #             #     rmse_min = err[1]
    #             #     bestmod = (int(leaf), depth, int(m))
    #             #     best_err = err
    # print(df)
    # df.to_csv('svd_vs_kmeans')
    # print(bestmod, best_err)


def grid_search_lasso(test_data, red):  # best sol
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)

    eM = EvaluateModel(train.env, red, 'lasso', red_dim=5, **{'load_red': True})
    eM.mod.reduce.load(test_data.env.system)
    rmse_min = 999999999
    bestmod = None
    best_err = None
    for alpha in np.logspace(-10, 20):
        hparam = {
            'alpha': alpha,
        }
        eM.mod.meanPredictor = pr.get_prediction('lasso')(dim=eM.mod.reduce.dim, **hparam)
        # print(1)
        eM.train(train)
        # eM.pred=None
        err = [eM.rmsle(test, False).mean(), eM.rmse(test, False).mean(), eM.mae(test, False).mean(),
               eM.rmse_per(test, False).mean(), eM.r_squared(test, False).mean()]
        print(alpha, err, bestmod, best_err)
        if rmse_min > err[1]:
            rmse_min = err[1]
            bestmod = alpha
            best_err = err
    print(bestmod, best_err)


def grid_search_mlp(test_data, red):  # best sol
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)

    eM = EvaluateModel(train.env, red, 'MLP', red_dim=5, **{'load_red': True})
    eM.mod.reduce.load(test_data.env.system)
    rmse_min = 999999999
    bestmod = None
    best_err = None

    # for dropout in np.linspace(0,0.99,7):
    for _ in range(100):
        for learning_rate in np.logspace(-10, 0, 10):
            dropout = np.random.uniform()
            hparam = {
                'dropout': dropout,
                'noeuds': map(int, np.random.randint(5, 50, size=3) / (1 - dropout)),
                'learning_rate': learning_rate,
                'epochs': 50,
                'verb': 0
            }
            eM.mod.meanPredictor = pr.get_prediction('MLP')(dim=eM.mod.reduce.dim, **hparam)
            # print(1)
            eM.train(train)
            # eM.pred=None
            err = [eM.rmsle(test, False).mean(), eM.rmse(test, False).mean(), eM.mae(test, False).mean(),
                   eM.rmse_per(test, False).mean(), eM.r_squared(test, False).mean()]
            p = (dropout, list(hparam['noeuds']), learning_rate)
            print(p, err, bestmod, best_err)
            if rmse_min > err[1]:
                rmse_min = err[1]
                bestmod = p
                best_err = err
    print(bestmod, best_err)


# def ry_one(test_data, red, pred, red_dim=5, comb=False):
#     train = test_data.get_partialdata_per(0, 0.85)
#     test = test_data.get_partialdata_per(0.85, 1)
#
#     eM = EvaluateModel(train.env, red, pred, red_dim=red_dim, **{'load_red': True,'is_combined':comb})
#     # t = eM.mod.train(train, **{'verb': 2})
#     t=eM.mod.train(train)
#     eM.mod.save()
#     ti = time.time()
#     p = eM.mod.predict(train)
#     tpred = time.time() - ti
#
#     # eM.mod.reduce.train(test)
#     # eM.mod.reduce.save(test_data.env.system)
#     # eM.mod.reduce.load(test_data.env.system)
#     # hparam = {
#     #     'n_estimators': 300,
#     #     'lr': 0.5,
#     #     'subsample': 1,
#     #     'max_depth': 19,
#     # }
#     # eM.mod.meanPredictor = pr.get_prediction(pred)(dim=eM.mod.reduce.dim, **hparam)
#     # print(1)
#     # eM.train(train)
#     # eM.pred=None
#     err = [eM.rmsle(test, False).mean(), eM.rmse(test, False).mean(), eM.mae(test, False).mean(),
#            eM.rmse_per(test, False).mean(), eM.r_squared(test, False).mean(), t, tpred]
#     del eM, test, train
#     print(err)
#     return err


def vote_for_pred(test_data, comb=False):
    train = test_data.get_partialdata_per(0, 0.85)
    test = test_data.get_partialdata_per(0.85, 1)
    best_err = np.ones(len(test.get_stations_col(2015))) * 10000
    votes = np.ones(best_err.shape, dtype=int) * -1
    best_err_2 = np.ones(best_err.shape) * 10000
    votes_2 = np.ones(best_err.shape, dtype=int) * -1
    red_dim = {
        'id': 1,
        'arr-dep': 2,
        'sum': 1,
        'svd': 10,
        'pca': 5,
        'autoencoder': 10,
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
        'id',
        'sum',
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
        'linear',
        'gbt',
        # 'ridge',
        # 'lasso',
        # 'MLP2',
        'MLP',
        # 'LSTM',
        'mean',
        'randforest',
        'decisiontree',
    ]
    i = 0
    l = []
    if comb:
        eM = EvaluateModel(train.env, None, None, red_dim=10, **{'load_red': True, 'is_combined': comb})
        eM.load_or_train(train)
        pp = eM.mod.predict(test)
        err = (pp - test.get_miniOD([])[test.get_stations_col(None)])[test.get_stations_col(2015)] ** 2
        err = err.as_matrix()
        err = err.mean(axis=0)

        votes_2[best_err > err] = votes[best_err > err]
        best_err_2[best_err > err] = best_err[best_err > err]

        votes_2[(best_err_2 > err) * (best_err <= err)] = i
        best_err_2[(best_err_2 > err) * (best_err <= err)] = err[(best_err_2 > err) * (best_err <= err)]

        votes[best_err > err] = i
        best_err[best_err > err] = err[best_err > err]
        l.append(('comb', 'comb'))
        i += 1
    for s in red:
        for p in pred:
            if ((s!='id') or (p!='randforest')):
                eM = EvaluateModel(train.env, s, p, red_dim=red_dim[s], **{'load_red': True})
                eM.train(train)
                eM.save()
                # eM.load_or_train(train)
                pp = eM.mod.predict(test)
                err = (pp - test.get_miniOD([])[test.get_stations_col(None)])[test.get_stations_col(2015)] ** 2
                err = err.as_matrix()
                err = err.mean(axis=0)

                votes_2[best_err > err] = votes[best_err > err]
                best_err_2[best_err > err] = best_err[best_err > err]

                votes_2[(best_err_2 > err) * (best_err <= err)] = i
                best_err_2[(best_err_2 > err) * (best_err <= err)] = err[(best_err_2 > err) * (best_err <= err)]

                votes[best_err > err] = i
                best_err[best_err > err] = err[best_err > err]
                l.append((s, p))
                i += 1
    unique, counts = np.unique(votes, return_counts=True)
    unique2, counts2 = np.unique(votes_2, return_counts=True)
    xx2 = np.zeros(len(red) * len(pred) + 1-1)
    xx2[unique] = counts
    xx = np.zeros(len(red) * len(pred) + 1-1)
    xx[unique] = counts
    xx[unique2] += counts2
    plt.figure(figsize=(10,8))
    plt.subplots_adjust(bottom=0.35)
    plt.bar(range(len(xx)), xx)
    plt.bar(range(len(xx2)), xx2)
    l1 = list(map(lambda x: x[0] + ' ' + x[1], l))
    plt.xticks(range(len(l1)), l1, rotation='vertical')
    if comb:
        plt.savefig('votes_comb.pdf', bbox_inches='tight')
    else:
        plt.savefig('votes.pdf', bbox_inches='tight')
    plt.show()


def feature_importance(test_data):
    import config
    train = test_data.get_partialdata_per(0, 0.85)
    # test = test_data.get_partialdata_per(0.85, 1)
    # eM = EvaluateModel(train.env, 'svd', 'decisiontree', red_dim=5, **{'load_red': True})
    # eM.train(train)
    # l = eM.mod.meanPredictor.DT.feature_importances_
    # plt.title('feature importance \n Decision Tree')
    # plt.bar(range(len(l)),l)
    # plt.xticks(range(len(l)),config.learning_var,rotation='vertical')
    # plt.show()
    # eM = EvaluateModel(train.env, 'svd', 'randforest', red_dim=5, **{'load_red': True})
    # eM.train(train)
    # l = eM.mod.meanPredictor.RF.feature_importances_
    # plt.title('feature importance \n Random Forest')
    # plt.bar(range(len(l)),l)
    # plt.xticks(range(len(l)),config.learning_var,rotation='vertical')
    # plt.show()
    eM = EvaluateModel(train.env, 'svd', 'gbt', red_dim=5, **{'load_red': True})
    eM.train(train)
    l = eM.mod.meanPredictor.model_GBT1[3].feature_importances_
    plt.title('feature importance \n Gradient boosted Tree')
    plt.bar(range(len(l)), l)
    plt.xticks(range(len(l)), config.learning_var, rotation='vertical')
    plt.show()
    l = eM.mod.meanPredictor.model_GBT1[4].feature_importances_
    plt.title('feature importance \n Gradient boosted Tree')
    plt.bar(range(len(l)), l)
    plt.xticks(range(len(l)), config.learning_var, rotation='vertical')
    plt.show()
    # l = eM.mod.meanPredictor.model_GBT1[2].feature_importances_
    # plt.title('feature importance \n Gradient boosted Tree')
    # plt.bar(range(len(l)),l)
    # plt.xticks(range(len(l)),config.learning_var,rotation='vertical')
    # plt.show()


def fnjpsdr(d):
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
        'id',
        'sum',
        'kmeans',
        'average',
        'complete',
        'weighted',
        'dep-arr',
        'GM',
        # 'SC',
        # 'maxkcut',
        # 'ODkmeans',
    ]
    pred = [
        'linear',
        'gbt',
        'ridge',
        'lasso',
        # 'MLP2',
        'MLP',
        # 'LSTM',
        'mean',
        'randforest',
        'decisiontree',
    ]
    f = open('errs', 'w')
    f.close()
    for s in red:
        for p in pred:
            with open('errs', 'a') as f:
                try:
                    r = red_dim[s]
                except KeyError:
                    r = 10
                res = test_one(d, s, p, red_dim=r)
                line = s + ',' + p + ','
                for i in res:
                    line += str(i) + ','
                line = line[:-1] + '\n'
                f.write(line)

                # grid_search_ridge(d, 'svd')
                # grid_search_gbt(d, 'svd')
                # ry_one(d,'svd','gbt')


def svd_dim():
    scores1 = pd.read_csv('D:\maitrise\svd_vs_precision.csv')
    scores2 = pd.read_csv('D:\maitrise\kmeans_vs_precision2.csv')
    id = {
        'rmsle': 0.468,
        'rmse': 1.262,
        'mae': 0.842,
        'rmse norm': 0.839,
        'r2': 0.571,
        'LL': -1.112
    }
    i = 1
    for ch in scores1.columns.values:
        if ch != 'dim':
            plt.subplot(2, 3, i)
            plt.plot(scores1['dim'], scores1[ch], linestyle="-", marker='x', label='svd')
            plt.plot(scores2['dim'], scores2[ch], linestyle="-", marker='x', label='kmeans')
            plt.hlines(id[ch], xmax=50, xmin=1, label='id')
            plt.legend()
            plt.title(ch)
            i += 1
    plt.show()

def corr():
    from config import root_path
    import pandas as pd
    df = pd.read_csv(root_path+'svd_vs_kmeans')
    print(df.corr().loc[['kmeans','svd','GM','autoencoder'],['kmeans','svd','GM','autoencoder']])

def best():
    from config import root_path
    import pandas as pd
    df = pd.read_csv(root_path+'svd_vs_kmeans')
    print('kmeans')
    print(df.loc[df['kmeans'].argmin()])
    print('svd')
    print(df.loc[df['svd'].argmin()])
    print('svd')
    print(df.loc[df['GM'].argmin()])
    print(df.loc[df['autoencoder'].argmin()])

if __name__ == '__main__':
    from preprocessing.Data import Data
    # ud = Environment('Bixi', 'train')
    d = Data(second_name='train')
    # grid_search_rf(d)

    vote_for_pred(d,comb=3)
    # for i in range(1,6):
    # ry_one(d,'autoencoder','randforest',5,False)
