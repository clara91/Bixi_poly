from model_station.EvaluateModels import *
from model_station.CombinedModelStation import CombinedModelStation
import config


if __name__ == '__main__':
    var = []
    var += [
        'Mois',
        'Heure',
        'Jour',
        # 'total',
        'Annee',
        'wday',
        # 'LV',
        # 'MMJ',
        # 'SD',
        # 'h0',
        # 'h1',
        # 'h2',
        # 'h3',
        # 'h4',
        # 'h5',
        # 'h6',
        # 'h7',
        # 'h8',
        # 'h9',
        # 'h10',
        # 'h11',
        # 'h12',
        # 'h13',
        # 'h14',
        # 'h15',
        # 'h16',
        # 'h17',
        # 'h18',
        # 'h19',
        # 'h20',
        # 'h21',
        # 'h22',
        # 'h23'
    ]
    meteo=[
        'temp',
        'vent',
        'precip',
        'visi',
        'averses',
        'pression',
        'fort',
        'nuageux',
        'brr',
        'Hum',
        'neige',
        'pluie',
        'verglas',
        'bruine',
        'poudrerie',
        'brouillard',
        'orage',
    ]
    # var+=meteo
    config.learning_var=var

    system = 'capitalBS'
    # system = 'citibike'
    # system = 'Bixi'

    # recompute_all_files('train')
    # recompute_all_files('test')
    data_train = Data(Environment(system, 'train')).get_partialdata_per(0, 1)
    # print(data_train.miniOD.shape)
    data_test = Data(Environment(system, 'test')).get_partialdata_per(0,1)
    # data_test = Data(Environment(system, 'test')).get_partialdata_per(0, 1)
    print(data_test.get_miniOD([]).shape)

    # mask = build_mask(data_test)
    # data_test = Data(Environment(system, 'train')).get_partialdata_per(0, 1)
    # print(data_test.miniOD.shape)
    red_dim = {
        'svd': 5,
        'svdevol': 5,
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
        # 'svd',
        # 'kmeans',
        # 'GM',
        'svdevol',
        # 'pca',
        # 'id',
        # 'sum',
        # 'average',
        # 'complete',
        # 'weighted',
        # 'dep-arr',
        # 'SC',
        # 'maxkcut',
        # 'ODkmeans',
    ]
    pred = [
        # 'linear',
        'gbt',
        # 'randforest',
        # 'decisiontree',
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
        # 'meanhour',
    ]
    pred_algos = pred * len(red)
    red_algos = [t for t in red for _ in pred]
    # pred_algos = ['gbt','randforest','gbt','randforest','randforest','gbt']
    # red_algos = ['svd','svd','kmeans','kmeans','id','sum']
    # hparam = {
    #     'var':False,
    #     'load_red': False,
    #     'is_model_station': True,
    #     'log': False,
    #     'mean': False,
    #     'norm': False,
    #     'load': False,
    #     'obj': 'mse',
    #     'hours': [],
    #     'decor':False,
    #     'is_combined': 3,
    #     'zero_prob': False,
    #     'red': {},
    #     'pred': {},
    # }
    # # train models
    # mods = EvaluatesModels(data_train, pred_algos, red_algos, red_dim, False, **hparam)
    # mods.names = [str(mods.algo[i][0]) + ' ' + str(mods.algo[i][1]) for i in range(len(mods.algo))]
    # # train combined
    # build_combined_model_rmse(data_train.get_partialdata_per(0,1), mods, pred_algos, red_algos, **hparam)
    # # quit()
    hparam = {
        'load_red': False,
        'is_model_station': True,
        'log': False,
        'mean': False,
        'norm': False,
        'load': False,
        'obj': 'mse',
        'hours': [],
        'decor': False,
        'is_combined': 0,
        'red':{},
        'pred':{},
        'n_week':4,
        'zero_prob': False,
        'var':False,
        'second_prob':{
            'pred':'linear',

        }
        # 'n_estimators': 130,
        # 'loss': 'ls',
        # 'subsample': 1,
        # 'lr': 0.1,
        # 'max_depth': 10
    }
    print(hparam['is_combined'])
    d = pd.DataFrame(columns=['station','size','algo','rmsle','rmse','mae','rmse_norm','r2','llP','dev','mpe','mape','train_time','test_time'])
    # for i in range(2,50,1):#,[1],[1,2,3],[1,2,3,4,5,6],[1,2,3,4,5,6,24],[1,2,3,4,5,6,24,25,26],[3,6,24],[3,6,24,48],[24,48],[24,48,72],range(25),range(49),range(73)]:
    #     print(i)
    #     red_dim['kmeans']=i
        # hparam['hours']=i
    # for i in :
    with tf.device('/cpu:0'):#['Start date 6063','End date 6063']['Start date '+str(s),'End date '+str(s)]
        # s = range(908)
        # s = 5005
        # s = 6221
        # for n_w_start in range(1,430,5):
        for n_w in np.linspace(0.1,10,50):
            print(n_w,"/10")
            hparam['n_week']=n_w
            from model_station.ModelStations import ModelStations
            # hparam['load']=False
            # mod = ModelStations(data_test.env,'svdevol','randforest',**hparam)
            # mod.load()
            # mod.train_inv(data_test.get_partialdata_n(n_w_start,168*hparam['n_week']))
            # mod.save()
            # hparam['load']=True


            # hparam['n_week']=n_w
            st=[]
            # for s in [6088,6924,6004,6193,6268,6436,6189,6061,6109,6230]:
            #     st.append('Start date '+str(s))
            #     st.append('End date '+str(s))
            # st = data_test.get_stations_col()
            df = compare_model(data_train, data_test, pred_algos, red_algos, red_dim, stations=st,**hparam)
            # df['station'] = n_w_start
            df['algo']=df.index
            df['weeks']=hparam['n_week']
            d=d.append(df,ignore_index=True)
    d=d[['weeks','station','size','algo','rmsle','rmse','mae','r2','llP','dev','mpe','mape','train_time','test_time']]
    d.to_csv('svdevol5.csv')
    print(d)
    # complete_analysis(data_train, data_test, load=False, norm=False)
    # plot_one_day(data_train, data_test, pred[0], red[0], load=True)