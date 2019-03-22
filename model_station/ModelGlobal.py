import model_station.Prediction as pr
import model_station.Reduction as reduce
from utils.modelUtils import maxi
from model_station.ModelStations import ModelStations


class ModelGlobal(ModelStations):
    """
    TO BE UPDATED
    """
    def __init__(self, env, reduction_method, prediction_method, **kwargs):
        super(ModelGlobal, self).__init__(env, reduction_method, prediction_method, dim=10, **kwargs)
        self.varPredictor = pr.get_prediction('linear')(2, kwargs=kwargs)
        self.to_global = reduce.get_reduction('dep-arr')(env)

    def get_objectives(self, learn):
        miniOD = learn
        return self.to_global.transform(miniOD)

    def train_variance(self, kwargs, learn):
        WH = self.get_factors(learn)
        # train variance on old data
        res = self.meanPredictor.predict(WH)
        res = self.reduce.inverse_tranform(res)
        res = self.to_global.transform(res)
        # print(res)
        WH = self.get_var_factors(learn)
        e = (maxi(res, 0.01) - self.get_objectives(learn)) ** 2
        self.varPredictor.train(WH, e, **kwargs)

    def predict(self, x):
        xt = self.get_factors(x)
        pred1 = self.meanPredictor.predict(xt, )  # self.reduce.get_factors(x))
        if self.norm:
            pred1 = pred1 * self.reduce.var / 5 + self.reduce.mean
        pred = self.reduce.inverse_tranform(pred1)
        pred = self.to_global.transform(pred)
        # print(pred.shape)
        return maxi(0, pred)

    def variance(self, x):
        xt = self.get_factors(x)
        pred = self.predict(x)
        var = maxi(self.varPredictor.predict(xt, ), 0.01)
        var = maxi(pred - pred ** 2, var)
        return var

    def get_y(self, x, since=None):
        return self.to_global.get_y(x, since)
