import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as GBT
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.linear_model import LinearRegression

import config
from utils.modelUtils import maxi, rmse, r_squared, mape, mae, rmsle
from model_station.ModelStations import ModelStations
from preprocessing.Data import Data


def add_pre_hours(data, mod, red_dim, tw, hh, boost):
    red = mod.reduce.transform(data)
    pred = mod.meanPredictor.predict(mod.get_factors(data))
    d = data.get_miniOD([])[config.learning_var]
    cols = list(d.columns.values)
    for h in range(1, tw + 1):
        for i in range(red_dim):
            cols.append('red' + str(i) + '_' + str(h))
    df = pd.DataFrame(np.zeros((d.shape[0] - tw - hh + 1, len(cols))), columns=cols)
    df[d.columns.values] = d.to_numpy()[tw + hh - 1:, :]
    for h in range(1, tw + 1):
        cols = []
        for i in range(red_dim):
            cols.append('red' + str(i) + '_' + str(h))
        if boost in [1]:  # learn on error
            df[cols] = red[(tw - h) + hh - 1:-h, :] - pred[(tw - h) + hh - 1:-h, :]
        elif boost in [2, 3]:  # learn 1st error then error evolution
            if h == 1:
                df[cols] = red[(tw - h) + hh - 1:-h, :] - pred[(tw - h) + hh - 1:-h, :]
                temp = red[(tw - h) + hh - 1:-h, :] - pred[(tw - h) + hh - 1:-h, :]
            else:
                df[cols] = red[(tw - h) + hh - 1:-h, :] - pred[(tw - h) + hh - 1:-h, :] - temp
        else:  # learn on previous hour trips
            df[cols] = red[(tw - h) + hh - 1:-h, :]
    return red, pred, df


def score(data, test, tw, hh, boost, mod):
    red, pred, mat = add_pre_hours(data, mod, 10, tw, hh, boost)
    randforest = []

    gbt = False
    if gbt:
        for i in range(red_dim):
            print(i)
            if boost == 0:
                randforest.append(GBT(n_estimators=200, max_depth=10, learning_rate=0.1))
                randforest[i].fit(mat.to_numpy(), red[tw + hh - 1:, i])
            elif boost == 1 or boost == 3:
                randforest.append(GBT(n_estimators=200, max_depth=7, learning_rate=0.1))
                cols = [] + config.learning_var  # ['Heure','wday']
                for h in range(tw):
                    cols.append('red' + str(i) + '_' + str(h + 1))
                randforest[i].fit(mat[cols].to_numpy(), red[tw + hh - 1:, i])
            elif boost == 2:
                randforest.append(GBT(n_estimators=50, max_depth=5, learning_rate=0.1))
                cols = [] + config.learning_var  # ['Heure','wday']
                for h in range(tw):
                    cols.append('red' + str(i) + '_' + str(h + 1))
                red_pred = red[tw + hh - 1:, i] - pred[tw + hh - 1:, i]
                randforest[i].fit(mat[cols].to_numpy(), red_pred)
    else:
        if boost == 2:
            randforest = RF(n_estimators=500, max_depth=20, min_samples_leaf=20)
            randforest.fit(mat.to_numpy(), red[tw + hh - 1:, :] - pred[tw + hh - 1, :])
        else:
            randforest = RF(n_estimators=95, max_depth=25, min_samples_leaf=10)
            randforest.fit(mat.to_numpy(), red[tw + hh - 1:, :])

    red_test, pred_t, mat_test = add_pre_hours(test, mod, 10, tw, hh, boost)
    if gbt:
        pred_test = np.zeros((mat_test.shape[0], red_dim))
        if boost == 0:
            for i in range(red_dim):
                pred_test[:, i] = randforest[i].predict(mat_test.to_numpy())
        else:
            for i in range(red_dim):
                cols = [] + config.learning_var
                # print(len(cols))
                for h in range(tw):
                    cols.append('red' + str(i) + '_' + str(h + 1))
                if boost==2:
                    pred_test[:, i] = randforest[i].predict(mat_test[cols].to_numpy())+pred_t[(tw + hh - 1):, i]
                else:
                    pred_test[:, i] = randforest[i].predict(mat_test[cols].to_numpy())
    else:
        pred_test = randforest.predict(mat_test)
    pred1 = mod.reduce.inv_transform(pred_test)
    pred_2 = maxi(pred1, 0.01)
    print('horizon', hh, 'tw', tw, boost)
    res_rmsle = rmsle(pred_2, test.get_miniOD([])[test.get_stations_col()].to_numpy()[tw + hh - 1:, :], axis=1)
    res_rmsle_base = rmsle(mod.predict(test), test.get_miniOD([])[test.get_stations_col()].to_numpy(), axis=1)
    res_rmse = rmse(pred_2, test.get_miniOD([])[test.get_stations_col()].to_numpy()[tw + hh - 1:, :], axis=1)
    res_rmse_base = rmse(mod.predict(test), test.get_miniOD([])[test.get_stations_col()].to_numpy(), axis=1)
    res_mae = mae(pred_2, test.get_miniOD([])[test.get_stations_col()].to_numpy()[tw + hh - 1:, :], axis=1)
    res_mae_base = mae(mod.predict(test), test.get_miniOD([])[test.get_stations_col()].to_numpy(), axis=1)
    res_r2 = r_squared(pred_2, test.get_miniOD([])[test.get_stations_col()].to_numpy()[tw + hh - 1:, :], axis=1)
    res_r2_base = r_squared(mod.predict(test), test.get_miniOD([])[test.get_stations_col()].to_numpy(), axis=1)
    res_mape = mape(pred_2, test.get_miniOD([])[test.get_stations_col()].to_numpy()[tw + hh - 1:, :], axis=1)
    res_mape_base = mape(mod.predict(test), test.get_miniOD([])[test.get_stations_col()].to_numpy(), axis=1)
    # f = open('time_boosting'+str(boost)+'.csv','w')
    m = np.array([res_rmsle, res_rmsle_base[tw:], res_rmse, res_rmse_base[tw:], res_mae, res_mae_base[tw:], res_r2,
                  res_r2_base[tw:], res_mape, res_mape_base[tw:]])
    df = pd.DataFrame(m.T, columns=['rmsle', 'rmsle_base', 'rmse', 'rmse_base', 'mae', 'mae_base', 'r2',
                            'r2_base', 'mape', 'mape_base'])
    print(rmsle(pred_2, test.get_miniOD([])[test.get_stations_col()].to_numpy()[tw + hh - 1:, :]))
    print(rmsle(mod.predict(test), test.get_miniOD([])[test.get_stations_col()].to_numpy()))
    print(rmse(pred_2, test.get_miniOD([])[test.get_stations_col()].to_numpy()[tw + hh - 1:, :]))
    print(rmse(mod.predict(test), test.get_miniOD([])[test.get_stations_col()].to_numpy()))
    print(mae(pred_2, test.get_miniOD([])[test.get_stations_col()].to_numpy()[tw + hh - 1:, :]))
    print(mae(mod.predict(test), test.get_miniOD([])[test.get_stations_col()].to_numpy()))
    print(mape(pred_2, test.get_miniOD([])[test.get_stations_col()].to_numpy()[tw + hh - 1:, :]))
    print(mape(mod.predict(test), test.get_miniOD([])[test.get_stations_col()].to_numpy()))
    print(r_squared(pred_2, test.get_miniOD([])[test.get_stations_col()].to_numpy()[tw + hh - 1:, :]))
    print(r_squared(mod.predict(test), test.get_miniOD([])[test.get_stations_col()].to_numpy()))
    return df


# for hh in range(1,100,10):

# print(hh)
if __name__ == '__main__':
    data = Data(second_name='train').get_partialdata_per(0, 1)
    test_set='jazz'
    if test_set=='sept':
        test = Data(second_name='train').get_partialdata_per(0, 1)
        mOD = test.get_miniOD([])
        i0 = mOD[(mOD['Mois'] == 9) * (mOD['Annee'] == 2016) * (mOD['Jour'] == 8)].index[0]
        test = test.get_partialdata_n(i0, 7 * 24)
    elif test_set=='jazz':
        test = Data(second_name='train').get_partialdata_per(0, 1)
        mOD = test.get_miniOD([])
        i0 = mOD[(mOD['Mois'] == 6) * (mOD['Annee'] == 2016) * (mOD['Jour'] == 29)].index[0]
        test = test.get_partialdata_n(i0, 11 * 24)
    elif test_set=='valid':
        test = Data(second_name='train').get_partialdata_per(0.8, 1)
    else:# test_set=='test':
        test = Data(second_name='test')
    print(test.get_miniOD([]).shape)

    hparam = {
        'norm': False,
        'hours': [],
        'load_red': False,
        'log': False,
        'mean': False,
        'decor': False,
        'red': {},
        'pred': {}
    }
    red_dim = 10
    mod = ModelStations(data.env, 'svd', 'randforest', red_dim, **hparam)
    mod.train(data)
    # for tw in range(24):
    tw = 6
    hor = 1
    # l=[]
    # for hor in range(1,15):
    #     df = score(data, test, tw, hor, 1, mod)
    #     df.index = df['score'].apply(lambda x:x+'_'+str(tw))
    #     l.append(df)

    df1 = score(data, test, tw, hor, 0, mod)
    df2 = score(data, test, tw, hor, 1, mod)
    df3 = score(data, test, tw, hor, 2, mod)
    df4 = score(data, test, tw, hor, 3, mod)
    l=[df1.T,df2.T,df3.T,df4.T]
    df = pd.concat(l)
    df.T.to_csv('time_boosting' +test_set+ str(tw) + '.csv')
    # score(1)
    # score(2)
    # score(3)
    # score(4)
    # Parallel(n_jobs=8)(score(hh) for hh in range(1,100,10))
