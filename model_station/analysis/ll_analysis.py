import joblib
import matplotlib.pyplot as plt
import numpy as np

from config import root_path
from model_station.EvaluateModels import EvaluatesModels
from model_station.EvaluateModels import *
from model_station.CombinedModelStation import CombinedModelStation

def compare_model(training_data, test_data, pred_algos, red_algos, red_dim, **kwargs):
    hparam = {
        'is_model_station': True,
        'norm': False,
        'load': True,
        'hours': [],
        'log': False,
        'is_combined': 3
    }
    hparam.update(kwargs)
    p = pred_algos * len(red_algos)
    s = [t for t in red_algos for _ in pred_algos]
    pred_algos = p
    red_algos = s

    mods = EvaluatesModels(training_data, pred_algos, red_algos, red_dim, hparam['is_combined'], **hparam)
    print(mods.algo)
    mods.names = [str(mods.algo[i][0]) + ' ' + str(mods.algo[i][1]) for i in range(len(mods.algo))]
    # mods.train(training_data)
    # mods.save()
    # analyse_loglike_distrib(test_data,mods,'P')
    # analyse_loglike_distrib(test_data,mods,'NB')
    # analyse_loglike_distrib(test_data,mods,'ZI')
    analyse_loglike(test_data, mods)


def analyse_kmeans_results(test_data):
    mods = EvaluatesModels(test_data, ['gbt'], ['kmeans'], {'kmeans': 10}, 0)
    LL = mods.compute_log_likelihood(test_data, 'P')
    rmse = mods.compute_rmse(test_data)
    mods.models[0].mod.reduce.load('Bixi')
    labels = mods.models[0].mod.reduce.labels
    centroid = []
    d = test_data.get_miniOD([],2015)[test_data.get_stations_col(2015)]
    for i in np.unique(labels):
        centroid.append(d.iloc[:, labels == i].mean(axis=1))
    dist = []
    for i in range(len(labels)):
        dist.append((np.array(d.iloc[:, i] - centroid[labels[i]]) ** 2).mean())
    plt.figure(1)
    plt.scatter(LL, dist)
    plt.xlabel('Loglikelihood P')
    plt.title('relationship between loglikelihood and distance to the centroid')
    plt.ylabel('distance to centroid')
    plt.figure(2)
    plt.scatter(rmse, dist)
    plt.xlabel('rmse')
    plt.title('relationship between rmse and distance to the centroid')
    plt.ylabel('distance to centroid')
    a = mods.models[0].mod.get_y(test_data.get_miniOD([])).mean(axis=0)
    plt.figure(3)
    plt.scatter(a,rmse)
    plt.show()


def corr_kmeans_svd(test_data):
    mods = EvaluatesModels(test_data, ['gbt'], ['svd'], {'svd': 5}, 0)
    LLsvd = mods.compute_log_likelihood(test_data, 'P')
    rmsesvd = mods.compute_rmse(test_data)
    mods.models[0].mod.load('Bixi')
    red = mods.models[0].mod.reduce
    distsvd = ((red.get_y(test_data.get_miniOD([],2015)) - red.inv_transform(red.transform(test_data))) ** 2).mean(axis=0)
    mods = EvaluatesModels(test_data, ['gbt'], ['kmeans'], {'kmeans': 10}, 0)
    LLkm = mods.compute_log_likelihood(test_data, 'P')
    rmsekm = mods.compute_rmse(test_data)
    mods.models[0].mod.load('Bixi')
    d = test_data.get_miniOD([],2015)[test_data.get_stations_col(2015)]
    labels = mods.models[0].mod.reduce.labels
    centroid = []
    for i in np.unique(labels):
        centroid.append(d.iloc[:, labels == i].mean(axis=1))
    distkm = []
    for i in range(len(labels)):
        distkm.append((np.array(d.iloc[:, i] - centroid[labels[i]]) ** 2).mean())
    print('LL corr', np.corrcoef(LLsvd, LLkm))
    print('rmse corr', np.corrcoef(rmsesvd, rmsekm))
    print('dist corr', np.corrcoef(distkm,distsvd))


def analyse_svd_results(test_data):
    mods = EvaluatesModels(test_data, ['gbt'], ['svd'], {'svd': 5}, 0)
    LL = mods.compute_log_likelihood(test_data, 'P')
    rmse = mods.compute_rmse_per(test_data)
    mods.models[0].mod.reduce.load('Bixi')
    red = mods.models[0].mod.reduce

    dist = ((red.get_y(test_data.get_miniOD([],2015)) - red.inv_transform(red.transform(test_data))) ** 2).mean(axis=0)
    plt.figure(1)
    plt.scatter(LL, dist)
    plt.xlabel('Loglikelihood P')
    plt.title('relationship between loglikelihood and approximation of svd')
    plt.ylabel('approximation')
    plt.figure(2)
    plt.scatter(rmse, dist)
    plt.xlabel('rmse')
    plt.title('relationship between rmse and approximation of svd')
    plt.ylabel('approximation')
    a = mods.models[0].mod.get_y(test_data.get_miniOD([])).mean(axis=0)
    plt.figure(3)
    plt.scatter(rmse,a)
    plt.xlabel('rmse')
    plt.title('relationship between rmse and size of station')
    plt.ylabel('size of station')
    plt.show()

def averageLL(test_data,mods):
    l1 = mods.names
    loglikeZI = np.array(
        joblib.load(root_path + '/LL_ZI'))  # np.array(mods.compute_log_likelihood_zero_inflated_poisson(test_data))
    loglikeP = np.array(joblib.load(root_path + '/LL_P'))  # np.array(mods.compute_log_likelihood_poisson(test_data))
    loglikeNB = np.array(
        joblib.load(root_path + '/LL_NB'))
    l = [np.array(l1),loglikeP.mean(axis=1),loglikeNB.mean(axis=1),loglikeZI.mean(axis=1)]
    print(np.array(l))

def analyse_loglike(test_data, mods):
    l1 = list(map(lambda x: x + ' NB', mods.names))
    l1.extend(list(map(lambda x: x + ' ZI', mods.names)))
    l1.extend(list(map(lambda x: x + ' P', mods.names)))
    loglikeZI = np.array(
        joblib.load(root_path + '/LL_ZI'))  # np.array(mods.compute_log_likelihood_zero_inflated_poisson(test_data))
    loglikeP = np.array(joblib.load(root_path + '/LL_P'))  # np.array(mods.compute_log_likelihood_poisson(test_data))
    loglikeNB = np.array(
        joblib.load(root_path + '/LL_NB'))  # np.array(mods.compute_log_likelihood_negative_binomial(test_data))
    # loglikeG = np.array(mods.compute_log_likelihood_gaussian(test_data))
    # loglikegeo = np.array(mods.compute_log_likelihood_geom(test_data))
    b=test_data.get_miniOD([])[test_data.get_stations_col(None)].sum()!=0
    loglikeNB=loglikeNB[:, b]
    loglikeZI=loglikeZI[:,b]
    loglikeP=loglikeP[:,b]
    LL = np.zeros((loglikeNB.shape[0] * 3, b.sum()))
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
    NLL = np.array(LL)
    r1 = np.argmax(NLL, axis=0)
    NLL[r1, range(NLL.shape[1])] = -np.inf
    r2 = np.argmax(NLL, axis=0)
    NLL[r2, range(LL.shape[1])] = -np.inf
    r3 = np.argmax(NLL, axis=0)
    # LL /= mx
    print('mean_best', np.mean(np.ma.masked_invalid(LL[r1, range(LL.shape[1])])))
    mx = np.max(LL, axis=0)
    LL = LL / mx
    means = test_data.get_miniOD([],None)[test_data.get_stations_col(None)].to_numpy()[:,b].mean(axis=0)
    # for i in np.unique(r):
    #     print(means[r == i].max())
    print('mean NB', means[r1 < loglikeNB.shape[0]].mean())
    print('mean ZI', means[(r1 < 2 * loglikeNB.shape[0]) * (r1 > loglikeNB.shape[0])].mean())
    print('mean poisson', means[(r1 < 3 * loglikeNB.shape[0]) * (r1 > 2 * loglikeNB.shape[0])].mean())
    # s = loglikeNB.shape[0]
    # print('vote1')
    # rd1 = r1[means<0.5]
    # rd2 = r2[means<0.5]
    # rd3 = r3[means<0.5]
    # rd1 = (rd1<2*s)*(rd1>=s)
    # rd2 = (rd2<2*s)*(rd2>=s)
    # rd3 = (rd3<2*s)*(rd3>=s)
    # print(np.mean(1-((1-rd1)*(1-rd2)*(1-rd3))))
    # print('ZI 0.5', ((r1 < 2 * loglikeNB.shape[0]) * (r1 > loglikeNB.shape[0]))[means<0.5].mean())
    # print('ZI 0.4', ((r1 < 2 * loglikeNB.shape[0]) * (r1 > loglikeNB.shape[0]))[means<0.4].mean())
    # print('ZI 0.3', ((r1 < 2 * loglikeNB.shape[0]) * (r1 > loglikeNB.shape[0]))[means<0.3].mean())
    # print('NB 2',(r1 < loglikeNB.shape[0])[means>2].mean())
    # print('NB 2.5',(r1 < loglikeNB.shape[0])[means>2.5].mean())
    # print('vote2')
    # print('ZI 0.5', ((r2 < 2 * loglikeNB.shape[0]) * (r2 > loglikeNB.shape[0]))[means<0.5].mean())
    # print('ZI 0.4', ((r2 < 2 * loglikeNB.shape[0]) * (r2 > loglikeNB.shape[0]))[means<0.4].mean())
    # print('ZI 0.3', ((r2 < 2 * loglikeNB.shape[0]) * (r2 > loglikeNB.shape[0]))[means<0.3].mean())
    # print('NB 2',(r2 < loglikeNB.shape[0])[means>2].mean())
    # print('NB 2.5',(r2 < loglikeNB.shape[0])[means>2.5].mean())
    # print('vote3')
    # print('ZI 0.5', ((r3 < 2 * loglikeNB.shape[0]) * (r3 > loglikeNB.shape[0]))[means<0.5].mean())
    # print('ZI 0.4', ((r3 < 2 * loglikeNB.shape[0]) * (r3 > loglikeNB.shape[0]))[means<0.4].mean())
    # print('ZI 0.3', ((r3 < 2 * loglikeNB.shape[0]) * (r3 > loglikeNB.shape[0]))[means<0.3].mean())
    # print('NB 2',(r3 < loglikeNB.shape[0])[means>2].mean())
    # print('NB 2.5',(r3 < loglikeNB.shape[0])[means>2.5].mean())

    # print('mean ga', means[(r < 4 * llzi.shape[0]) * (r > 3 * llzi.shape[0])].mean())
    # print('mean Gaussian', means[r > 3 * loglikeNB.shape[0]].mean())
    unique1, counts1 = np.unique(r1, return_counts=True)
    unique2, counts2 = np.unique(r2, return_counts=True)
    unique3, counts3 = np.unique(r3, return_counts=True)
    v1 = np.zeros(3 * len(mods.names))
    v1[unique1] = counts1
    v2 = np.zeros(3 * len(mods.names))
    v2[unique2] = counts2
    v3 = np.zeros(3 * len(mods.names))
    v3[unique3] = counts3
    print('model name,v1,v2,v3,max,min, mean trips per model, LL/maxLL, N inf')
    for i in range(LL.shape[0]):
        a = means[r1==i]
        if a.any():
            m=a.min()
            M=a.max()
        else:
            m=-1
            M=-1
        print(l1[i],v1[i],v2[i],v3[i],M,m, a.mean(), np.mean(np.ma.masked_invalid(LL[i, :])), np.sum(np.isinf(LL[i, :])),np.ma.corrcoef(np.ma.masked_invalid(LL[i, :]), means[:LL.shape[1]])[1, 0], sep=',')
    v2 += v1
    v3 += v2
    l=len(mods.names)
    print('NB',v1[:l].sum(),v2[:l].sum(),v3[:l].sum())
    print('ZI',v1[l:2*l].sum(),v2[l:2*l].sum(),v3[l:2*l].sum())
    print('P',v1[2*l:3*l].sum(),v2[2*l:3*l].sum(),v3[2*l:3*l].sum())
    plt.figure(figsize=(20,10))
    plt.bar(range(3 * len(mods.names)), v3, label='third vote')
    plt.bar(range(3 * len(mods.names)), v2, label='second vote')
    plt.bar(range(3 * len(mods.names)), v1, label='first vote')
    # plt.hist(r1, bins=np.arange(-0.5, 3 * len(mods.names) + 1, 1))

    # l1.extend(list(map(lambda x: x + ' geo', mods.names)))
    # l1.extend(list(map(lambda x: x + ' G', mods.names)))
    plt.legend()
    plt.xticks(range(len(l1)), l1, rotation=270)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('distrib_votes.pdf',bbox_inches='tight')
    plt.show()

    # for m in mods.loglike:
    #     print(m)
    #     print(m[np.logical_not(np.isinf(m))].mean())


def analyse_loglike_distrib(test_data, mods, distrib):
    ll = mods.compute_log_likelihood(test_data, distrib, axis=0)
    joblib.dump(ll, root_path + '/LL_' + distrib)
    r = np.argmax(np.array(ll), axis=0)
    means = test_data.get_miniOD([],None)[test_data.get_stations_col(None)].mean(axis=0).to_numpy()
    for i in np.unique(r):
        print(means[r == i].max())
    plt.hist(r, bins=np.arange(-0.5, len(mods.names), 1))
    l1 = mods.names[:]
    plt.title(distrib)
    plt.xticks(range(len(mods.names)), l1, rotation='vertical')
    plt.show()

    for m in ll:
        print(m)
        print(m[np.logical_not(np.isinf(m))].mean())


if __name__ == '__main__':
    from preprocessing.Data import Data
    from preprocessing.Environment import Environment
    import tensorflow as tf

    data_train = Data(Environment('Bixi', 'train')).get_partialdata_per(0, 1)
    data_test = Data(Environment('Bixi', 'test')).get_partialdata_per(0, 1)


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

    pred = [
        'gbt',
        # 'linear',
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
        # 'randforest',
        # 'decisiontree',
    ]
    red = [
        'svd',
        # 'autoencoder',
        # 'pca',
        # 'id',
        # 'sum',
        # 'kmeans',
        # 'average',
        # 'complete',
        # 'weighted',
        # 'dep-arr',
        # 'GM',
        # 'SC',
        # 'maxkcut',
        # 'ODkmeans',
    ]

    hparam = {
        'load_red': False,
        'is_model_station': True,
        'log': False,
        'mean': False,
        'norm': False,
        'load': True,
        'obj': 'mse',
        'hours': [],
        'decor': False,
        'is_combined': 0,
        'red':{},
        'pred':{},
        'n_week':4,
        'zero_prob':False,
        'var':True,
        'second_prob':{
            'pred':'linear',

        }
    }
    pred_algos = pred * len(red)
    red_algos = [t for t in red for _ in pred]
    with tf.device('/cpu:0'):
        mods = EvaluatesModels(data_train, pred_algos, red_algos, red_dim, False, **hparam)
        mods.names = [str(mods.algo[i][0]) + ' ' + str(mods.algo[i][1]) for i in range(len(mods.algo))]
        # train combined
        # build_combined_model_rmse(data_train.get_partialdata_per(0,1), mods, pred_algos, red_algos)
        hparam['load']=True
        compare_model(data_train, data_test, pred, red, red_dim, **hparam)
