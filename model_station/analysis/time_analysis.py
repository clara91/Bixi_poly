
import model_station.Reduction as S
import model_station.Prediction as P
from model_station.ModelStations import ModelStations
from model_station.CombinedModelStation import CombinedModelStation
from preprocessing.Environment import Environment
from preprocessing.Data import Data
import time

def redution_time():
    data = Data(Environment('Bixi','train'))
    red_dim = {
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
        'id':-1,
        'sum':1,
        'dep-arr':2,
        'random':10,
    }
    train={}
    transform = {}
    inv_transform = {}
    for n in S.algorithms.keys():
        start_time = time.time()
        red=S.get_reduction(n)(data.env,dim=red_dim[n])
        red.train(data)
        end_train_time=time.time()
        t = red.transform(data)
        end_transform_time = time.time()
        red.inv_transform(t)
        end_inv_transform = time.time()
        train[n]=end_train_time-start_time
        transform[n]=end_transform_time-end_train_time
        inv_transform[n]=end_inv_transform-end_transform_time
    print(train)
    print(transform)
    print(inv_transform)
    return  train,transform,inv_transform

def model_time():
    file = open('model_time.csv', 'w')
    data = Data(Environment('Bixi','train'))
    red_dim = {
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
        'id':-1,
        'sum':1,
        'dep-arr':2,
        'random':10,
    }
    train={}
    transform = {}
    inv_transform = {}
    red_dim = {
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
        'id':-1
    }

    red = [
        # 'autoencoder',
        'svd',
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
    pred = [
        # 'linear',
        # 'gbt',
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
        'randforest',
        # 'decisiontree',
    ]
    for n in red:
        for p in pred:
            file = open('model_time.csv', 'a')
            start_time = time.time()
            mod = ModelStations(data.env,n,p,dim=red_dim[n])
            mod.train(data,var=False)
            end_train_time=time.time()
            mod.predict(data)
            end_transform_time = time.time()
            train[n+' '+p]=end_train_time-start_time
            transform[n+' '+p]=end_transform_time-end_train_time
            file.write(n+' '+p+', '+str(train[n+' '+p])+', '+str(transform[n+' '+p])+'\r')
            file.close()
    print(train)
    print(transform)
    print(inv_transform)
    return  train,transform,inv_transform

def combined_time():
    data = Data(Environment('Bixi', 'train'))
    red_dim = {
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
        'id': -1,
        'sum': 1,
        'dep-arr': 2,
        'random': 10,
    }
    start_time = time.time()
    mod = CombinedModelStation(data.env,**{'is_combined':3})
    mod.train(data)
    end_train_time=time.time()
    mod.predict(data)
    end_transform_time = time.time()
    print('combined',end_train_time-start_time,end_transform_time-end_train_time)
    return end_train_time-start_time,end_transform_time-end_train_time


if __name__ == '__main__':
    model_time()