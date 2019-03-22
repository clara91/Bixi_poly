import joblib

from config import root_path
from model_station.ModelStations import *


class CombinedModelStation(object):
    """
    class to build an ensemble model
    """
    def __init__(self, env, **kwargs):
        self.env=env
        self.hparam = {
            'var':False,        #learn variance predictor
            'norm': False,      #normalize objectives
            'hours': [],
            'load_red': False, #load reduction part
            'is_combined': 1,   #number of model to take per station
            'second_pred':{     #parameters of second predictor
                'pred':'linear',

            },
            'zero_prob':False,  #learn a zero probability predictor
            'red': {},          #reduction hyperparam
            'pred': {}          #prediction hyperparam
        }

        self.hparam.update(kwargs)
        # print(self.hparam['is_combined'])
        self.reduction_methods = joblib.load(root_path + 'model_station/prediction_models/combined_model/red'+env.system)
        self.prediction_methods = joblib.load(root_path + 'model_station/prediction_models/combined_model/pred'+env.system)
        self.dims = joblib.load(root_path + 'model_station/prediction_models/combined_model/dims'+env.system)
        self.stations_best = np.zeros((self.hparam['is_combined'], joblib.load(
            root_path + 'model_station/prediction_models/combined_model/best_0'+env.system).shape[0]))
        for k in range(self.hparam['is_combined']):
            self.stations_best[k, :] = joblib.load(
                root_path + 'model_station/prediction_models/combined_model/best_' + str(k)+env.system)
        print(self.reduction_methods, self.prediction_methods)
        self.models = []
        k = 0
        modhparam = self.hparam.copy()
        modhparam['var']=False
        for red, pred in zip(self.reduction_methods, self.prediction_methods):
            mod = ModelStations(env, red, pred, self.dims[k], **self.hparam)
            mod.indice = k
            mod.is_use = (k in self.stations_best)
            self.models.append(mod)
            k += 1
        self.name = 'c' + str(self.hparam['is_combined']) +' c'+str(self.hparam['is_combined'])# str(self.reduction_methods) + ' ' + str(self.prediction_methods)
        self.secondPredictor = pr.get_prediction(self.hparam['second_pred']['pred'])(dim=len(Data(env).get_stations_col(None)),
                                                                                     **self.hparam['second_pred'])

    def train(self, data, **kwargs):
        starttime = time.time()
        for mod in self.models:
            if mod.is_use:
                mod.train(data, **kwargs)
                mod.save()
        if self.hparam['zero_prob']:
            self.train_zero_prob(data,**self.hparam['pred'])
        elif self.hparam['var']:
            self.train_variance(data, **self.hparam['pred'])
        print('training time', 'combined combined', time.time() - starttime)

    def train_variance(self, learn, **kwargs):
        t =self.hparam['var']
        self.hparam['var']=False
        self.load()
        self.hparam['var']=t
        # train variance on old data
        WH = self.get_factors(learn)
        res = self.predict(WH)
        WH = self.get_var_factors(learn)
        e = (maxi(res, 0.01) - self.get_objectives(learn)) ** 2
        # print(self.get_objectives(learn))

        self.secondPredictor.train(WH, e, **kwargs)

    def train_zero_prob(self, learn, **kwargs):
        # WH = self.get_var_factors(learn)
        # res = self.predict(WH)
        # e = (maxi(res, 0.01) - self.get_objectives(learn)) ** 2
        obj = self.get_objectives(learn)==0
        WH = self.get_factors(learn)
        self.secondPredictor.train(WH, obj, **kwargs)


    # extract learning information
    def get_all_factors(self, learn):
        return self.models[0].get_all_factors(learn)

    def get_factors(self, learn):
        return self.models[0].get_factors(learn)

    def get_var_factors(self, learn):
        return self.models[0].get_var_factors(learn)

    def predict(self, x):
        starttime = time.time()
        predictions = []
        for mod in self.models:
            if mod.is_use:
                pred = mod.predict(x)
                predictions.append(pred)
            else:
                predictions.append(np.array([]))
        shape = pred.shape
        res = np.zeros(shape)
        # for i in self.stations_best:
        #     print(predictions[i].shape)
        for k in range(self.hparam['is_combined']):
            tab = self.stations_best[k, :]
            tab = tab.astype(int)
            for i in np.unique(tab):
                res[:, tab == i] += predictions[i][:, tab == i]
        res /= self.hparam['is_combined']

        print('prediction time', 'combined combined', time.time() - starttime)
        return res

    def variance(self, x):
        xt = self.get_var_factors(x)
        pred = self.predict(x)
        var = maxi(self.secondPredictor.predict(xt), 0.01)
        var = maxi(pred - pred ** 2, var)
        return var

    def zero_prob(self, x):
        xt = self.get_factors(x)
        p = self.predict(x)
        res=maxi(self.secondPredictor.predict(xt),np.exp(-p))
        return mini(maxi(res, 0.01), 0.99)

    def save(self):
        for mod in self.models:
            mod.hparam['var']=False
            mod.save()
        self.secondPredictor.save(add_path='comb' + self.env.system)
            # save model per station
            # save list of models

    def load(self):
        for mod in self.models:
            t = mod.hparam['var']
            mod.hparam['var']=False
            mod.load()
            mod.hparam['var']=t
        if self.hparam['var']:
            self.secondPredictor.load(add_path='comb' + self.env.system)


    def load_or_train(self, data, **kwargs):
        for mod in self.models:
            t = mod.hparam['var']
            mod.hparam['var']=False
            mod.load_or_train(data)
            mod.hparam['var']=t
        if self.hparam['var']:
            try :
                self.secondPredictor.load(add_path="comb" + self.env.system)
            except IOError:
                self.secondPredictor.train(data)

    def reset(self):
        for mod in self.models:
            mod.reset()

    def get_objectives(self, learn):
        return self.models[0].get_objectives(learn)

    def get_y(self, x, since=None):
        return self.models[0].get_y(x, since)
