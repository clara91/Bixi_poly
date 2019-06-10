import joblib
import numpy as np
from pandas import DataFrame
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering as SC
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.mixture import GaussianMixture as GM

from config import root_path
from preprocessing.Data import Data
from preprocessing.Station import Stations
import utils.modelUtils as utils


def loc(self, add_path=''):
    """
    generate the path where the model should be saved/loaded
    :param self: the redution model
    :param add_path: complementary path
    :return: path (string)
    """

    directory = 'model_station/'
    dim_red = 'dim_red/'
    OD = 'OD/'
    clustering = 'clustering/'
    if self.dim_red:
        return root_path + directory + dim_red + self.algo + str(self.dim) + add_path
    elif self.OD:
        return root_path + directory + OD + self.algo + str(self.dim) + add_path
    else:
        return root_path + directory + clustering + self.algo + str(self.dim) + add_path

"""
all Reduction function and classes. They should extend the Reduction class and reimplement its functions
"""
class Reduction(object):
    def __init__(self, env, dim=20, log=False, mean=False):
        """
        initialisation, dim is the number of dimension of destinaiton space
        :param env: a Environment object with the location of desired objects
        :param dim: dimension of destination space
        """
        self.preselect = Data(env).get_stations_col(None)
        # self.preselect_2015 = Data(env).get_stations_col(2015)
        # self.preselect_2017 = Data(env).get_stations_col(2017)
        self.OD = False
        self.dim_red = False
        self.log = log
        self.mean = mean

    def get_y(self, x:DataFrame, since=None):
        """
        get the column form the dataframe x corresponding to the reduced part
        :param x: Dataframe of trips
        :param since: if 2015 all 2015 and 2016 stations, if 2017 or None, all stations
        :return: numpy matrix with desired columns
        """
        return x[self.preselect]


    def load(self, add_path=''):
        """
            load the redution inplace
            :param add_path: complement in the path
            :return: None
            """
        # raise NotImplementedError('train not Implemented')
        self.location = loc(self, add_path)
        #C:/Users/Clara Martins/Documents/Doutorado/Pierre Code/Bixi_poly/model_station/dim_red/svd10Bixi
        s = joblib.load(self.location)
        for attr in dir(s):
            if not '__' in attr:
                setattr(self, attr, getattr(s, attr))

    def train(self, data:Data, **kwargs):
        """
        train the redution on the given data (inplace)
        :param data: a Data object
        :param kwargs: extra parameters  
        :return: None 
        """
        raise NotImplementedError('train not Implemented')

    def transform(self, x:Data):
        """
        do the redution
        :param x: Dataframe
        :return: numpy matrix 
        """
        raise NotImplementedError('get_factors not Implemented')

    def inv_transform(self, y:np.array):
        """
        inverse the redution
        :param y: numpy matrix
        :return: numpy matrix
        """
        raise NotImplementedError('inv_transform not Implemented')

    def inverse_tranform(self, y:np.array):
        """
        inverse the redution
        :param y: numpy matrix
        :return: Dataframe
        """
        res = self.inv_transform(y)
        return DataFrame(res, columns=self.preselect)

    # save the model
    def save(self, add_path=''):
        """
        save the model 
        :return: None
        """
        self.location = loc(self, add_path)
        joblib.dump(self, self.location)
        # raise NotImplementedError('save not Implemented')

    def reset(self):
        raise NotImplementedError('reset Not Implemented')

class DimRed(Reduction):
    def __init__(self, env, log=False, mean=False, *args):
        super(DimRed, self).__init__(env, log=log, mean=mean)
        self.dim_red = True

    def reset(self):
        del self.model

class DimRedSum(DimRed):
    def __init__(self, env, dim=20, log=False, mean=False, *args):
        super(DimRedSum, self).__init__(env, log=log, mean=mean)
        self.dim = 1
        self.algo = 'sum'
        self.location = loc(self)

    def train(self, data, **kwargs):
        self.model = None
        if self.mean:

            _, self.mean_coef = data.get_miniOD(hours=[], log=self.log, mean=self.mean)[
                self.preselect].to_numpy().sum(axis=1)
            self.mean_coef = self.mean_coef[self.preselect].to_numpy()

    def transform(self, data):
        if self.mean:
            x, _ = data.get_miniOD(hours=[], log=self.log, mean=self.mean)[self.preselect].to_numpy().sum(
                axis=1)
        else:
            x = data.get_miniOD(hours=[], log=self.log, mean=self.mean)[self.preselect].to_numpy().sum(axis=1)
        x = x.reshape((x.shape[0], 1))
        return x

    def inv_transform(self, y):
        res = np.zeros((y.shape[0], len(self.preselect)))
        res = ((res + y.reshape((y.shape[0], 1))) / len(self.preselect))
        if self.mean:
            res = res * self.mean_coef
        if self.log:
            res = np.exp(res)
        return res

class DimRedIdentity(DimRed):
    def __init__(self, env, dim=20, log=False, mean=False, *args):
        super(DimRedIdentity, self).__init__(env, log=log, mean=mean)
        self.algo = 'id'
        self.dim = len(self.preselect)
        self.location = loc(self)

    def train(self, data, **kwargs):
        if self.mean:
            _, self.mean_coef = data.get_miniOD(hours=[], log=self.log, mean=self.mean)[
                self.preselect].to_numpy()
            self.mean_coef = self.mean_coef[self.preselect].to_numpy()
        self.model = None

    def transform(self, data):
        if self.mean:
            res, _ = data.get_miniOD(hours=[], log=self.log, mean=self.mean)[self.preselect].to_numpy()
        else:
            res = data.get_miniOD(hours=[], log=self.log, mean=self.mean)[self.preselect].to_numpy()
        return res

    def inv_transform(self, y):
        if self.mean:
            y = y * self.mean_coef
        if self.log:
            return np.exp(y)
        else:
            return y

class DimRedDepArr(DimRed):
    def __init__(self, env, dim=2, log=False, mean=False, *args):
        super(DimRedDepArr, self).__init__(env, log=log, mean=mean)
        self.algo = 'dep-arr'
        self.dim = 2
        self.location = loc(self)
        self.dep = []
        self.arr = []
        for s in Stations(env, None).get_ids():
            self.arr.append('End date ' + str(s))
            self.dep.append('Start date ' + str(s))

    def train(self, data, **kwargs):
        self.model = None

    def transform(self, data):
        if isinstance(data, DataFrame):
            res = np.zeros((data.shape[0], 2))
            res[:, 1] = np.nansum(data[self.arr].to_numpy(), axis=1)
            res[:, 0] = np.nansum(data[self.dep].to_numpy(), axis=1)
            return res
        else:
            res = np.zeros((data.get_miniOD(hours=[], log=self.log, mean=self.mean).shape[0], 2))
            res[:, 1] = np.nansum(data.get_miniOD(hours=[], log=self.log, mean=self.mean)[self.arr].to_numpy(),
                                  axis=1)
            res[:, 0] = np.nansum(data.get_miniOD(hours=[], log=self.log, mean=self.mean)[self.dep].to_numpy(),
                                  axis=1)
            return res

    def get_y_dep_arr(self, x, since=None):
        return DataFrame(self.transform(x), columns=['Start date', 'End date'], index=x.index)

    def inv_transform(self, y):
        res = np.zeros((y.shape[0], len(self.preselect)))
        res[:, range(0, len(self.preselect), 2)] = (
                (res[:, range(0, len(self.preselect), 2)] + y[:, 0].reshape((y.shape[0], 1))) / len(self.arr))
        res[:, range(1, len(self.preselect), 2)] = (
                (res[:, range(0, len(self.preselect), 2)] + y[:, 1].reshape((y.shape[0], 1))) / len(self.dep))
        if self.mean:
            res = res * self.mean_coef
        if self.log:
            res = np.exp(res)
        return res

class DimRedPCA(DimRed):
    def __init__(self, env, dim=10, log=False, mean=False, *args):
        super(DimRedPCA, self).__init__(env, log=log, mean=mean)
        self.algo = 'pca'
        self.dim = dim
        self.location = loc(self)
        self.model = PCA(n_components=dim)

    def train(self, data, **kwargs):
        # print(self.preselect_2015)
        if self.mean:
            res, self.mean_coef = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
            self.mean_coef = self.mean_coef[self.preselect].to_numpy()
        else:
            res = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        self.model.fit(res[self.preselect].to_numpy())

    def transform(self, data):
        if self.mean:
            res, _ = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        else:
            res = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        x = self.model.transform(res[self.preselect].to_numpy())
        return x

    def inv_transform(self, y):
        res = np.zeros((y.shape[0], len(self.preselect)))
        res[:, :len(self.preselect)] = self.model.inverse_transform(y)
        if self.mean:
            res = res * self.mean_coef
        if self.log:
            res = np.exp(res)
        return res


    def reset(self):
        del self.model

class DimRedPCAKernel(DimRed):
    def __init__(self, env, dim=10, log=False, mean=False, **kwargs):
        hparam = {
            'kernel': 'poly'
        }
        hparam.update(kwargs)
        super(DimRedPCAKernel, self).__init__(env, log=log, mean=mean)
        self.algo = 'kpca'
        self.dim = dim
        self.location = loc(self)
        self.model = KernelPCA(kernel=hparam['kernel'], n_components=dim, n_jobs=-1, fit_inverse_transform=True)

    def train(self, data, **kwargs):
        if self.mean:
            res, self.mean_coef = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
            self.mean_coef = self.mean_coef[self.preselect].to_numpy()
        else:
            res = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        self.model.fit(res[self.preselect].to_numpy())

    def transform(self, data):
        if self.mean:
            res, _ = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        else:
            res = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        x = self.model.transform(res[self.preselect].to_numpy())
        return x

    def inv_transform(self, y):
        res = np.zeros((y.shape[0], len(self.preselect)))
        res[:, :len(self.preselect)] = self.model.inverse_transform(y)
        if self.mean:
            res = res * self.mean_coef
        if self.log:
            res = np.exp(res)
        return res

        # def load(self, add_path=''):
        #     self.location = loc(self, add_path)
        #     s = joblib.load(self.location)
        # self.MSnorm = s.MSnorm
        # self.mean = s.mean
        # self.var = s.var
        # self.model = s.model

    def reset(self):
        # del self.MSnorm
        # del self.mean
        # del self.var
        del self.model

class DimRedSVD(DimRed):
    def __init__(self, env, dim=10, log=False, mean=False, *args):
        super(DimRedSVD, self).__init__(env, log=log, mean=mean)
        # self.objetives=['dim'+str(i) for i in range(dim)]
        self.algo = 'svd'
        self.dim = dim
        self.location = loc(self)
        self.model = TruncatedSVD(n_components=dim)

    def train(self, data, **kwargs):
        if self.mean:
            res, self.mean_coef = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
            self.mean_coef = self.mean_coef[self.preselect].to_numpy()
        else:
            res = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        d = res[self.preselect].to_numpy()
        self.model.fit(d)

    def transform(self, data):
        if self.mean:
            res, _ = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        else:
            res = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        x = self.model.transform(res[self.preselect].to_numpy())
        return x

    def inv_transform(self, y):
        res = y
        # if res.shape[1]==1:
        if len(res.shape)==1:
            res = res.reshape((res.shape[0],1))
        res = self.model.inverse_transform(res)
        # else:
        #     res = self.model.inverse_transform(res)
        if self.mean:
            res = res * self.mean_coef
        if self.log:
            res = np.exp(res)
        return res

class DimRedEvolvingSVD(DimRedSVD):
    def __init__(self, env, dim=10, log=False, mean=False, *args):
        super(DimRedEvolvingSVD, self).__init__(env, log=log, mean=mean)
        # self.objetives=['dim'+str(i) for i in range(dim)]
        self.algo = 'svdevol'
        self.location = loc(self)
        self.mat=None
        self.dim=dim

    def train_inv(self, x:np.matrix, y:np.matrix):
        from sklearn.linear_model import LinearRegression,OrthogonalMatchingPursuit
        lr = LinearRegression(n_jobs=-1,fit_intercept=False)
        lr.fit(x,y)
        # OMP = OrthogonalMatchingPursuit(n_nonzero_coefs=2,normalize=False)
        # OMP.fit(x,y)
        self.mat = lr.coef_

    def inv_transform(self, y):
        res = y
        # if res.shape[1]==1:
        if len(res.shape)==1:
            res = res.reshape((res.shape[0],1))
        res = np.dot(y,self.mat.T)
        # else:
        #     res = self.model.inverse_transform(res)
        if self.mean:
            res = res * self.mean_coef
        if self.log:
            res = np.exp(res)
        return res

class DimRedAutoEncoder(DimRed):
    def __init__(self, env, dim=10, log=False, mean=False, **kwargs):
        DimRed.__init__(self, env, log=log, mean=mean)
        # self.objetives=['dim'+str(i) for i in range(dim)]
        self.algo = 'autoencoder'
        self.dim = dim
        self.location = loc(self)

        # self.model = Model()
        self.encoder = None
        self.decoder = None
        self.model = None
        self.define_model(**kwargs)

    def save(self, add_path=''):
        self.location = loc(self, add_path)
        # joblib.dump(self.log, self.location + 'log')
        # joblib.dump(self.log, self.location + 'mean')
        # joblib.dump(self.log, self.location + '')
        # joblib.dump(self.mean, self.location + 'mean')
        # joblib.dump(self.var, self.location + 'var')
        # joblib.dump(self.MSnorm, self.location + 'norm')
        temp1 = self.model
        temp2 = self.encoder
        temp3 = self.decoder
        self.encoder.save(self.location + 'encoder')
        # self.model.model.save(self.location + 'model')
        self.decoder.save(self.location + 'decoder')
        self.encoder = 1
        self.decoder = 2
        self.model = 3
        DimRed.save(self, add_path)
        # super(DimRedAutoEncoder, self).save(add_path)
        self.model = temp1
        self.encoder = temp2
        self.decoder = temp3
        # self.model.model.save(self.location + 'model')

    def load(self, add_path=''):
        DimRed.load(self, add_path)
        from keras.models import load_model
        self.location = loc(self, add_path)
        # self.model = Model()
        # self.log = joblib.load(self.location + 'log')
        # self.MSnorm = joblib.load(self.location + 'norm')
        del self.encoder
        del self.decoder
        del self.model
        self.encoder = load_model(self.location + 'encoder')
        self.decoder = load_model(self.location + 'decoder')
        self.model = None
        # self.model.model = load_model(self.location + 'model')
        # self.model.model = load_model(self.location + 'model')
        # self.mean = joblib.load(self.location + 'mean')
        # self.var = joblib.load(self.location + 'var')

    def define_model(self, **kwargs):
        from keras import losses
        import keras.optimizers
        import keras.regularizers as regularizers
        from keras.layers import Input, Dense, Dropout, GaussianNoise, Activation, BatchNormalization
        from keras.models import Model
        hparams = {
            'std_noise': 0.2,
            'activation': 'relu',
            'noeuds': [70, 35],
            'dropout': 1,
            'learning_rate': 2,
            # 'noeuds': [250, 125, 70, 35, self.dim],

            'epsilon': 6.8129e-05,
            # 'learning_rate': 0.01,
            'reg_l1': 0,
            'reg_l2': 0,
            'beta_1': 0.977,
            'beta_2': 0.99973,
            'decay': 0,
            # 'n_couches': 5
        }
        hparams.update(kwargs)
        dim = len(self.preselect)
        input = Input(shape=(dim,))
        encoded = GaussianNoise(hparams['std_noise'])(input)
        encoded = Activation('relu')(encoded)
        # encoded = input
        ########################################
        # Encoder
        ########################################
        for i in hparams['noeuds']:
            # print(i)
            encoded = Dropout(hparams['dropout'])(encoded)
            encoded = Dense(i,
                            activation=hparams['activation'],
                            kernel_regularizer=regularizers.l1_l2(
                                hparams["reg_l1"],
                                hparams["reg_l2"]),
                            # kernel_initializer=init,
                            )(encoded)
            encoded = BatchNormalization()(encoded)
        # print(self.dim)
        encoded = Dense(self.dim,
                        # activation=hparams['activation'],
                        # kernel_regularizer=regularizers.l1_l2(
                        #     hparams["reg_l1"],
                        #     hparams["reg_l2"]),
                        # kernel_initializer=init,
                        )(encoded)
        self.encoder = Model(inputs=input, outputs=encoded)
        #########################################
        # Decoder
        #########################################
        encoded_input = Input(shape=(self.dim,))
        decoded = GaussianNoise(hparams['std_noise'])(encoded_input)
        # decoded = encoded_input
        for i in range(len(hparams['noeuds']) - 2, -1, -1):
            # print(hparams['noeuds'][i])
            decoded = Dropout(hparams['dropout'])(decoded)
            decoded = Dense(hparams['noeuds'][i],
                            activation=hparams['activation'],
                            kernel_regularizer=regularizers.l1_l2(
                                hparams["reg_l1"],
                                hparams["reg_l2"]),
                            # kernel_initializer=init,
                            )(decoded)
            encoded = BatchNormalization()(encoded)
        decoded = Dropout(hparams['dropout'])(decoded)
        decoded = Dense(dim,
                        activation=hparams['activation'],
                        kernel_regularizer=regularizers.l1_l2(
                            hparams["reg_l1"],
                            hparams["reg_l2"]),
                        # kernel_initializer=init,
                        )(decoded)
        self.decoder = Model(inputs=encoded_input, outputs=decoded)
        self.model = Model(inputs=input, outputs=self.decoder(self.encoder(input)))
        obj = losses.mean_squared_error
        opt = keras.optimizers.adadelta(lr=hparams['learning_rate'])
        self.model.compile(optimizer=opt, loss=obj, metrics=['mae', 'mse'])

    def train(self, data, **kwargs):
        args = {
            'n_epochs': 100,
            'batch_size': 30,
            'verb': 2
        }
        args.update(kwargs)
        if self.mean:
            res, self.mean_coef = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
            self.mean_coef = self.mean_coef[self.preselect].to_numpy()
        else:
            res = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        self.model.fit(res[self.preselect].to_numpy(),
                       res[self.preselect].to_numpy(),
                       epochs=args['n_epochs'],
                       batch_size=args['batch_size'],
                       verbose=args['verb'],
                       shuffle=True,
                       validation_split=0.15
                       )

    def transform(self, data):
        # print(self is copy_of_self)
        if self.mean:
            res, _ = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        else:
            res = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        x = self.encoder.predict(res[self.preselect].to_numpy())
        return x

    def inv_transform(self, y):
        res = y
        res = self.decoder.predict(res)
        if self.mean:
            res = res * self.mean_coef
        if self.log:
            res = np.exp(res)
        return res

class VariationalAutoEncoder(DimRedAutoEncoder):
    def __init__(self, env, dim=10, log=False, mean=False, *args):
        super(VariationalAutoEncoder, self).__init__(env, dim, log=log, mean=mean, *args)
        self.define_model()

    def define_model(self, **kwargs):
        from keras import losses
        import keras.optimizers
        import keras.regularizers as regularizers
        from keras.layers import Input, Dense, Dropout
        from keras.models import Model
        hparams = {
            'dropout': 0.01,
            'activation': 'relu',
            'noeuds': [250, 125, 70, 35, self.dim],
            'epsilon': 6.8129e-05,
            'learning_rate': 0.01,
            'reg_l1': 0,
            'reg_l2': 0,
            'beta_1': 0.977,
            'beta_2': 0.99973,
            'decay': 0.2,
            'n_couches': 3
        }
        hparams.update(kwargs)
        dim = len(self.preselect)
        input = Input(shape=(dim,))
        encoded = input
        ########################################
        # Encoder
        ########################################
        for i in hparams['noeuds']:
            # print(i)
            encoded = Dropout(hparams['dropout'])(encoded)
            encoded = Dense(i,
                            activation=hparams['activation'],
                            kernel_regularizer=regularizers.l1_l2(
                                hparams["reg_l1"],
                                hparams["reg_l2"]),
                            # kernel_initializer=init,
                            )(encoded)

        self.model.encoder = Model(inputs=input, outputs=encoded)
        #########################################
        # Decoder
        #########################################
        encoded_input = Input(shape=(hparams['noeuds'][-1],))
        decoded = encoded_input
        for i in range(len(hparams['noeuds']) - 2, -1, -1):
            # print(hparams['noeuds'][i])
            decoded = Dropout(hparams['dropout'])(decoded)
            decoded = Dense(hparams['noeuds'][i],
                            activation=hparams['activation'],
                            kernel_regularizer=regularizers.l1_l2(
                                hparams["reg_l1"],
                                hparams["reg_l2"]),
                            # kernel_initializer=init,
                            )(decoded)
        decoded = Dropout(hparams['dropout'])(decoded)
        decoded = Dense(dim,
                        activation=hparams['activation'],
                        kernel_regularizer=regularizers.l1_l2(
                            hparams["reg_l1"],
                            hparams["reg_l2"]),
                        # kernel_initializer=init,
                        )(decoded)
        self.model.decoder = Model(inputs=encoded_input, outputs=decoded)
        # creer un layer random
        self.model.model = Model(inputs=input, outputs=self.model.decoder(self.model.encoder(input)))
        obj = losses.mean_squared_error
        opt = keras.optimizers.adadelta()
        self.model.model.compile(optimizer=opt, loss=obj, metrics=['mae', 'mse'])

class Clustering(Reduction):
    def __init__(self, env, log=False, mean=False, dim=10):
        super(Clustering, self).__init__(env, log=log, mean=mean)
        self.dim_red = False
        self.dim = dim
        self.station_coef = None

    def pre_train(self, data, **kwargs):
        """
        operations to do before training
        :param data: a data object
        :param kwargs: not used
        :return: learning matrix
        """
        learn = data.get_synthetic_miniOD([], self.log)[self.preselect].to_numpy()
        # coef_2015 = data.get_miniOD(2015)[self.preselect_2015].to_numpy().mean(axis=0)
        coef = learn.mean(axis=0)
        coef = utils.maxi(coef,0.001)
        # self.station_coef_2015 = np.zeros(coef_2015.shape)
        self.station_coef = coef
        return learn / coef

    def post_train(self, data, **kwargs):
        """
        operations to do after training
        :param data: a Data object
        :param kwargs: not used
        :return: none
        """
        # s = np.zeros(len(self.preselect))
        # s[:len(self.preselect_2015)] = self.labels
        # d = data.get_synthetic_miniOD(2017)
        # r = np.zeros(shape=(d.shape[0], self.dim))
        # for i in range(self.dim):
        #     r[:, i] = d.to_numpy()[:, self.labels].mean(axis=1)
        # k = 0
        # self.labels = s

    def transform(self, data):
        """
        get the objectives using the trained redution
        :param data: complete pandas dataframe of trips 
        :return: new model objectives (numpy matrix) same number of rows, self.dim columns
        """
        # coef_2017 = data.get_synthetic_miniOD(None)[self.preselect].to_numpy().mean(axis=0)
        if self.mean:
            res, self.mean_coef = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
            self.mean_coef = self.mean_coef[self.preselect].to_numpy()
        else:
            res = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        data = res[self.preselect].to_numpy() / utils.maxi(self.station_coef,0.001)
        res = np.zeros((data.shape[0], self.dim))
        # self.station_coef_2015 = np.zeros(coef_2015.shape)
        # self.station_coef_2017 = np.zeros(coef_2017.shape)
        for i in range(self.dim):
            a = self.labels == i
            # a_2015 = a[:len(self.preselect_2015)]
            # self.station_coef_2015[a_2015] = coef_2015[a_2015] / coef_2015[a_2015].sum()
            # self.station_coef_2017[a] = coef_2017[a] / coef_2017[a].sum()
            res[:, i] = data[:, a].mean(axis=1)
        # print(coef_2017)
        return res

    def inv_transform(self, y):
        """
        Do the inverse transformation for clusters
        :param y: the data to inverse transform (numpy matrix)
        :return: the inversed data (numpy matrix)
        """
        res = np.zeros((y.shape[0], len(self.preselect)))
        for i in range(len(self.labels)):
            res[:, i] = y[:, self.labels[i]]
        # if self.B_norm:
        #     res = res * self.norm
        res = res * self.station_coef
        if self.mean:
            res = res * self.station_coef
        if self.log:
            res = np.exp(res)
        return res

        # def save(self, add_path=''):
        #     """
        #     save the model
        #     :param add_path:complementary path of this model
        #     :return: none
        #     """
        #     self.location = loc(self, add_path)
        #     joblib.dump(self, self.location)

        # def load(self, add_path=''):
        #     """
        #     load the model in place
        #     :param add_path: complementary path of this model
        #     :return: none
        #     """
        #     self.location = loc(self, add_path)
        #     s = joblib.load(self.location)
        #     self.station_coef_2015 = s.station_coef_2015
        # self.station_coef = s.station_coef
        # self.log = s.log
        # self.MSnorm = s.MSnorm
        # self.B_norm = s.B_norm
        # self.mean = s.mean
        # self.var = s.var
        # self.norm = s.norm
        # self.labels = s.labels

class ClusteringKmeans(Clustering):
    def __init__(self, env, dim=10, mean=False, log=False):
        super(ClusteringKmeans, self).__init__(env, dim=dim, log=log, mean=mean)
        self.algo = 'kmeans'
        self.location = loc(self)

    def train(self, data, **kwargs):
        learn = self.pre_train(data)

        # print(learn.shape)
        km = KMeans(n_clusters=self.dim, n_jobs=-1, n_init=100).fit(learn.T)
        self.labels = km.labels_
        self.post_train(data)

class RandomClustering(Clustering):
    def __init__(self, env, dim=10, mean=False, log=False):
        super(RandomClustering, self).__init__(env, dim=dim, log=log, mean=mean)
        self.algo = 'random'
        self.location = loc(self)

    def train(self, data, **kwargs):
        learn = self.pre_train(data)
        self.labels = np.random.randint(0, self.dim, len(self.preselect))
        self.post_train(data)

class ClusteringHierarchy(Clustering):
    def __init__(self, env, algo, log=False, mean=False, load=True, dim=10):
        super(ClusteringHierarchy, self).__init__(env, dim=dim, log=log, mean=mean)
        self.algo = algo
        self.location = loc(self)

    def train(self, data, **kwargs):
        self.pre_train(data)
        learn = data.get_synthetic_miniOD([], self.log)[self.preselect].to_numpy()
        # norm = learn.sum(axis=0)
        norm = self.station_coef
        l = np.zeros((learn.shape[0] + 1, learn.shape[1]))
        l[0, :] = norm
        l[1:, :] = learn / norm
        # print(l.shape)
        lin = linkage(
            l.T,
            metric='seuclidean',
            method=self.algo
        )
        self.labels = fcluster(lin, self.dim, criterion='maxclust')
        self.labels -= 1
        self.post_train(data)

    @staticmethod
    def distance(x, y):
        xn = x[0]
        # print(x.shape)
        yn = y[0]
        precision = x * (1 - x) / xn + y * (1 - y) / yn
        precision = np.array(list(map(lambda xx: max(1e-8, xx), precision)))
        alpha = 1 / np.abs((xn - yn) / precision)
        return 1 / np.sum(alpha)


class ClusteringAverage(ClusteringHierarchy):
    def __init__(self, env, load=True, log=False, mean=False, dim=10):
        super(ClusteringAverage, self).__init__(env, 'average', load=load, dim=dim, log=log,
                                                mean=mean)


class ClusteringComplete(ClusteringHierarchy):
    def __init__(self, env, load=True, mean=False, log=False, dim=10):
        super(ClusteringComplete, self).__init__(env, 'complete', load=load, dim=dim, log=log,
                                                 mean=mean)


class ClusteringWeighted(ClusteringHierarchy):
    def __init__(self, env, load=True, mean=False, log=False, dim=10):
        super(ClusteringWeighted, self).__init__(env, 'weighted', load=load, dim=dim, log=log,
                                                 mean=mean)


class ClusteringGM(Clustering):
    def __init__(self, env, dim=10, mean=False, log=False):
        super(ClusteringGM, self).__init__(env, dim=dim, log=log, mean=mean)
        self.algo = 'GM'
        self.location = loc(self)

    def train(self, data, **kwargs):
        learn = self.pre_train(data)
        gm = GM(n_components=self.dim).fit(learn.T)
        self.labels = gm.predict(learn.T)
        self.post_train(data)


class ClusteringSpectral(Clustering):
    def __init__(self, env, dim=10, mean=False, log=False):
        super(ClusteringSpectral, self).__init__(env, dim=dim, log=log, mean=mean)
        self.algo = 'GM'
        self.location = loc(self)

    def train(self, data, **kwargs):
        learn = self.pre_train(data)
        gm = SC(n_clusters=self.dim, affinity=self.distance, n_jobs=-1, n_init=100).fit(learn.T)
        # self.centers = km.cluster_centers_
        self.labels = gm.labels_
        self.post_train(data)

    def distance(self, x, y):
        xn = np.sqrt(np.sum(x * x))
        x_n = x / xn
        yn = np.sqrt(np.sum(y * y))
        y_n = y / yn
        precision = x_n * (1 - x_n) / xn + y_n * (1 - y_n) / yn
        precision = np.array(list(map(lambda xx: max(1e-8, xx), precision)))
        alpha = 1 / np.abs((xn - yn) / precision)
        return 1 / np.sum(alpha)


class ClusteringMaxKCut(Clustering):
    def __init__(self, env, dim=10, mean=False, log=False):
        super(ClusteringMaxKCut, self).__init__(env, dim=dim, log=log, mean=mean)
        self.algo = 'maxkcut'
        self.location = loc(self)

    def train(self, data, **kwargs):
        OD = data.get_OD()
        from utils.modelUtils import repeat_max_kcut
        self.labels = repeat_max_kcut(OD, self.dim, 30)
        self.post_train(data)


class ClusteringOD(Reduction):
    def __init__(self, env, dim=10, mean=False, norm=True, log=False):
        super(ClusteringOD, self).__init__(env, log=log, mean=mean)
        self.dim_red = False
        self.OD = True
        self.dim = dim * dim
        self.private_dim = dim
        self.B_norm = norm
        self.station_coef_arr = None
        self.station_coef_dep = None
        # self.location = loc(self)

    def post_train(self, data):
        self.station_coef_arr = np.zeros(data.get_OD().shape[0])
        self.station_coef_dep = np.zeros(data.get_OD().shape[0])
        OD = data.get_OD()
        dep = OD.sum(axis=1)
        arr = OD.sum(axis=0)
        for l in range(self.private_dim):
            self.station_coef_arr[self.labels == l] = arr[self.labels == l] / arr[self.labels == l].sum()
            self.station_coef_dep[self.labels == l] = dep[self.labels == l] / dep[self.labels == l].sum()

    # get_factors the objectives using the trained redution
    # input: complete pandas dataframe
    # output: new model objectives (numpy matrix) same number of rows
    def transform(self, data):
        da = data.get_hour_OD()
        hours = np.unique(list(map(lambda x: int(x[0]), da.index.values)))
        invhours = {}
        for i in range(len(hours)):
            invhours[hours[i]] = i
        invstations = {}
        for i in range(len(data.get_stations_ids(None))):
            invstations[data.get_stations_ids(None)[i]] = i
        if self.mean:
            mOD, _ = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        else:
            mOD = data.get_miniOD(hours=[], log=self.log, mean=self.mean)
        res = np.zeros((mOD.shape[0], self.private_dim, self.private_dim))
        da['ind'] = da.index.values
        d = da.to_numpy()
        # print(invstations)
        for i in range(da.shape[0]):
            l = d[i, :]
            h, s1, s2 = l[-1]
            # print(s1,s2)
            try:
                res[invhours[int(h)], self.labels[invstations[s1]], self.labels[invstations[s2]]] += l[0]
            except KeyError:
                pass
            except IndexError:
                pass
        res = res.reshape((mOD.shape[0], self.private_dim * self.private_dim))
        return res

    ## inverse get_factors the objectives
    # returns a numpy matrix
    def inv_transform(self, y):
        y = y.reshape((y.shape[0], self.private_dim, self.private_dim))
        rdep = np.zeros((y.shape[0], self.station_coef_dep.shape[0]))
        rarr = np.zeros((y.shape[0], self.station_coef_arr.shape[0]))
        for s in range(self.station_coef_arr.shape[0]):
            rdep[:, s] = y[:, self.labels[s], :].sum(axis=1) * self.station_coef_dep[s]
            rarr[:, s] = y[:, :, self.labels[s]].sum(axis=1) * self.station_coef_arr[s]

        result = np.zeros((y.shape[0], 2 * self.station_coef_arr.shape[0]))
        result[:, range(0, 2 * self.station_coef_arr.shape[0], 2)] = rarr
        result[:, range(1, 2 * self.station_coef_dep.shape[0], 2)] = rdep
        if self.log:
            result = np.exp(result)
        return result[:, :len(self.preselect)]

        # def save(self, add_path=''):
        #     self.location = loc(self, add_path)
        #     joblib.dump(self, self.location)

        # def load(self, add_path=''):
        #     self.location = loc(self, add_path)
        #     s = joblib.load(self.location)
        # self.B_norm = s.B_norm
        # self.norm = s.norm
        # self.log = s.log
        # self.labels = s.labels
        # self.station_coef_arr = s.station_coef_arr
        # self.station_coef_dep = s.station_coef_dep

    def reset(self):
        del self.B_norm
        del self.labels
        del self.station_coef_arr
        del self.station_coef_dep


class ClusteringODMaxKCut(ClusteringOD):
    def __init__(self, env, dim=10, mean=False, log=False):
        super(ClusteringODMaxKCut, self).__init__(env, dim=dim, norm=True, log=log, mean=mean)
        self.algo = 'maxkcut'
        self.location = loc(self)

    def train(self, data, **kwargs):
        OD = data.get_OD()
        from utils.modelUtils import repeat_max_kcut
        self.labels = repeat_max_kcut(OD, self.private_dim, 5)
        self.post_train(data)


class ClusteringODKmeans(ClusteringOD):
    def __init__(self, env, dim=10, mean=False, log=False):
        self.algo = 'kmeans'
        super(ClusteringODKmeans, self).__init__(env, dim=dim, norm=True, log=log, mean=mean)
        self.location = loc(self)

    def train(self, data, **kwargs):
        learn = data.get_OD()
        km = KMeans(n_clusters=self.private_dim, n_jobs=-1).fit(learn.T)
        self.labels = km.labels_
        self.post_train(data)


algorithms ={
    'svd':DimRedSVD,
    'svdevol':DimRedEvolvingSVD,
    'pca':DimRedPCA,
    'kpca':DimRedPCAKernel,
    'sum':DimRedSum,
    'kmeans':ClusteringKmeans,
    'average':ClusteringAverage,
    'complete':ClusteringComplete,
    'weighted':ClusteringWeighted,
    'random':RandomClustering,
    'dep-arr':DimRedDepArr,
    'GM':ClusteringGM,
    'id':DimRedIdentity,
    'autoencoder':DimRedAutoEncoder,
    # 'SC':ClusteringSpectral,
    # 'maxkcut':ClusteringODMaxKCut,
    # 'ODkmeans':ClusteringODKmeans,
}


def get_reduction(red):
    if isinstance(red, Reduction):
        return red
    elif isinstance(red, str):
        try:
            return algorithms[red]
        except KeyError:
            raise Exception("could not interprete " + red + " class")
    else:
        raise Exception("could not interprete class " + red)
