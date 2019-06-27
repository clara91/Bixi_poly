import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, nbinom, skellam

import config
from utils.modelUtils import mini, maxi, proba_sum, proba_diff


class ServiceLevel(object):
    def __init__(self, env, mod, arr_vs_dep=0.5):
        """
        initialize information for computing service level
        :param env: environement information (Environment object)
        :param mod: model of class ModelStation
        :param arr_vs_dep: float between 0 and 1, how to valuedeparture vs arrival.  
        """
        self.env = env
        self.mod = mod
        self.mean = None
        self.var = None
        self.dict = {}
        pre = 1e-10
        arr_vs_dep = min(1-pre,max(pre,arr_vs_dep))
        self.arr = arr_vs_dep
        self.dep = 1-arr_vs_dep

    def compute_mean_var(self, WT, data, predict):
        """
        compute and store estimated mean and variance
        :param WT: features to predict
        :param data: Data object with station information 
        :param predict: True: predict the mean, else use real data
        :return: None
        """
        try:
            self.dict['cols'] = self.mod.reduce.preselect 
        except AttributeError:
            self.dict['cols'] = self.mod.models[0].reduce.preselect
        self.dict['arr_cols'] = data.get_arr_cols(None)
        self.dict['stations'] = data.get_stations_ids(None)
        self.dict['capacities'] = data.get_stations_capacities(None).to_numpy().flatten()
        if predict:
            self.mean = pd.DataFrame(self.mod.predict(x = WT), columns=self.dict['cols'])[
                self.dict['cols']].to_numpy()
        else:
            if config.learning_var.__contains__('Heure'):
                dd = pd.merge(WT, data.get_miniOD(), 'left',
                              on=['Mois', 'Jour', 'Annee', 'Heure'])
            else:
                dd = pd.merge(WT, data.get_miniOD(), 'left',
                              on=['Mois', 'Jour', 'Annee', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9',
                                  'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21',
                                  'h22',
                                  'h23'])
            self.mean = dd[self.dict['cols']].to_numpy()
        self.var = pd.DataFrame(self.mod.variance(WT), columns=self.dict['cols'])[
            self.dict['cols']].to_numpy()
        self.var[self.var == 0] = 0.01

    def compute_service_gaussian(self, current_capacity=None):
        """
        compute the service level with a gaussian distribution
        :param current_capacity: current state of the network, if None compute the service for all capacities
        :return: service level : array if current_capacity set, else return a matrix with a service levels: each column 
            correspond to one station
        """
        cum_var = pd.DataFrame(np.cumsum(self.var, axis=0), columns=self.dict['cols'])
        cum_mean = pd.DataFrame(np.cumsum(self.mean, axis=0), columns=self.dict['cols'])
        m = pd.DataFrame(self.mean, columns=self.dict['cols'])
        arr = m[self.dict['arr_cols']].to_numpy()
        dep = m.drop(self.dict['arr_cols'], axis=1).to_numpy()
        for s in self.dict['stations']:
            cum_mean[str(s)] = cum_mean['End date ' + str(s)].to_numpy() - cum_mean[
                'Start date ' + str(s)].to_numpy()
            cum_var[str(s)] = cum_var['End date ' + str(s)].to_numpy() + cum_var[
                'Start date ' + str(s)].to_numpy()
        self.dict['cum_mean'] = cum_mean[list(map(str, self.dict['stations']))].to_numpy()
        self.dict['cum_var'] = cum_var[list(map(str, self.dict['stations']))].to_numpy()
        if current_capacity is None:
            service = np.zeros((np.max(self.dict['capacities'] + 1), dep.shape[1]))
            for c in range(np.max(self.dict['capacities']) + 1):
                cap = np.ones(dep.shape[1]) * c
                cum_mean = np.add(self.dict['cum_mean'], cap)
                proba_empty = norm.cdf(0, loc=cum_mean, scale=self.dict['cum_var'])
                proba_full = norm.sf(np.array(self.dict['capacities']), loc=cum_mean, scale=self.dict['cum_var'])
                service_loc = (dep * (1 - proba_empty)).sum(axis=0) / (np.sum(dep, axis=0) + 0.001)
                service_ret = (arr * (1 - proba_full)).sum(axis=0) / (np.sum(arr, axis=0) + 0.001)
                service[c] = 2 * mini(self.dep * service_loc, self.arr * service_ret)
                service[c, c > self.dict['capacities']] = 0
        else:
            cum_mean = np.add(self.dict['cum_mean'], current_capacity)
            proba_empty = norm.cdf(0, loc=cum_mean, scale=self.dict['cum_var'])
            proba_full = norm.sf(np.array(self.dict['capacities']), loc=cum_mean, scale=self.dict['cum_var'])
            service_loc = (dep * (1 - proba_empty)).sum(axis=0) / (np.sum(dep, axis=0) + 0.001)
            service_ret = (arr * (1 - proba_full)).sum(axis=0) / (np.sum(arr, axis=0) + 0.001)
            service = 2 * mini(self.dep * service_loc, self.arr * service_ret)
        return service

    def compute_service_level_from_proba_matrix(self, mat, available_bikes=None):
        """
        compute service level from the probability matrix, for any distribution (defined by the probability matrix)
        :param mat: matrix timewindow*n_stations*N, mat[t,s,k] is the proba that there are k departures (arrivals) at 
                time t at station s
        :param available_bikes: current network status, if None compute service for all statuses
        :return: service level : array if available_bikes set, else return a matrix with a service levels: each column 
            correspond to one station
        """
        res_mat = np.zeros(mat.shape)
        # compute the probability matrix of dep-arr for each station and each hour (cumulative)
        for t in range(mat.shape[0]):
            for i in range(mat.shape[1]):
                if t == 0:
                    res_mat[t, i, :] = mat[t, i, :]
                else:
                    p = proba_sum(res_mat[t - 1, i, :], mat[t, i, :])
                    p[res_mat.shape[2] - 1] = p[res_mat.shape[2] - 1:].sum()
                    res_mat[t, i, :] = p[:res_mat.shape[2]]
        p_dep_minus_arr = np.zeros((mat.shape[0], int(mat.shape[1] / 2), 2 * mat.shape[2] - 1))
        for s in range(int(mat.shape[1] / 2)):
            for t in range(mat.shape[0]):
                p_dep_minus_arr[t, s, :] = proba_diff(res_mat[t, 2 * s + 1, :], res_mat[t, 2 * s, :])
        # compute the probability of being superior or inferior or k (axis 2)
        proba_dep_inf = np.cumsum(p_dep_minus_arr, axis=2)
        proba_dep_sup = 1 - proba_dep_inf
        proba_dep_inf = proba_dep_inf - p_dep_minus_arr
        # get expected number of departure and arrivals
        m = pd.DataFrame(self.mean, columns=self.dict['cols'])
        arr = m[self.dict['arr_cols']].to_numpy()
        dep = m.drop(self.dict['arr_cols'], axis=1).to_numpy()
        if not (available_bikes is None):
            available_doks = self.dict['capacities'] - available_bikes
            available_doks = maxi(int(p_dep_minus_arr.shape[2] / 2) - available_doks, 0)
            available_bikes += int(p_dep_minus_arr.shape[2] / 2)
            available_bikes = mini(available_bikes, int(p_dep_minus_arr.shape[2]) - 1)
            available_bikes = available_bikes.astype(dtype=int)
            available_doks = available_doks.astype(dtype=int)
            # compute service level
            service_level_dep = (dep * proba_dep_inf[:, range(proba_dep_inf.shape[1]), available_bikes]).sum(
                axis=0) / dep.sum(axis=0)
            service_level_arr = (arr * proba_dep_sup[:, range(proba_dep_sup.shape[1]), available_doks]).sum(
                axis=0) / arr.sum(axis=0)
            service = 2 * mini(self.dep * service_level_dep, self.arr * service_level_arr)
        else:
            service = np.zeros((np.max(self.dict['capacities'] + 1), dep.shape[1]))
            for c in range(np.max(self.dict['capacities'])):
                available_bikes = mini(self.dict['capacities'], c)
                available_doks = self.dict['capacities'] - available_bikes
                available_doks = maxi(int(p_dep_minus_arr.shape[2] / 2) - available_doks, 0)
                available_bikes += int(p_dep_minus_arr.shape[2] / 2)
                available_bikes = mini(available_bikes, int(p_dep_minus_arr.shape[2]) - 1)
                available_bikes = available_bikes.astype(dtype=int)
                available_doks = available_doks.astype(dtype=int)
                # compute service level
                service_level_dep = (dep * proba_dep_inf[:, range(proba_dep_inf.shape[1]), available_bikes]).sum(
                    axis=0) / dep.sum(axis=0)
                service_level_arr = (arr * proba_dep_sup[:, range(proba_dep_sup.shape[1]), available_doks]).sum(
                    axis=0) / arr.sum(axis=0)
                service[c] = 2 * mini(self.dep * service_level_dep, self.arr * service_level_arr)
                service[c, c > self.dict['capacities']] = 0
        return service

    def compute_service_level_Poisson(self, available_bikes=None):
        # lambdas = np.cumsum(self.mean,axis=0)
        cum_mean = pd.DataFrame(np.cumsum(self.mean, axis=0), columns=self.dict['cols'])
        m = pd.DataFrame(self.mean, columns=self.dict['cols'])
        arr = m[self.dict['arr_cols']].to_numpy()
        dep = m.drop(self.dict['arr_cols'], axis=1).to_numpy()
        # for s in self.dict['stations']:
        #     cum_mean[str(s)] = cum_mean['End date ' + str(s)].to_numpy() - cum_mean[
        #         'Start date ' + str(s)].to_numpy()
        cum_arr = cum_mean[self.dict['arr_cols']]
        cum_dep = cum_mean.drop(self.dict['arr_cols'], axis=1)
        # self.dict['cum_mean'] = cum_mean[list(map(str, self.dict['stations']))].to_numpy()
    
        if available_bikes is None:
            service = np.zeros((np.max(self.dict['capacities'] + 1), dep.shape[1]))
            for c in range(np.max(self.dict['capacities']) + 1):
                cap = np.ones(dep.shape[1]) * c
                loc = cap
                # cum_mean = np.add(self.dict['cum_mean'], cap)
                proba_empty = skellam.cdf(0, mu1=cum_arr, mu2=cum_dep, loc=loc)
                proba_full = skellam.sf(np.array(self.dict['capacities']), mu1=cum_arr, mu2=cum_dep, loc=loc)
                service_loc = (dep * (1 - proba_empty)).sum(axis=0) / (np.sum(dep, axis=0) + 0.001)
                service_ret = (arr * (1 - proba_full)).sum(axis=0) / (np.sum(arr, axis=0) + 0.001)
                service[c] = 2 * mini(self.dep * service_loc, self.arr * service_ret)
                service[c, c > self.dict['capacities']] = 0
        else:
            loc = np.ones(cum_arr.shape) * available_bikes
            proba_empty = skellam.cdf(0, mu1=cum_arr, mu2=cum_dep, loc=loc)
            proba_full = skellam.sf(np.array(self.dict['capacities']), mu1=cum_arr, mu2=cum_dep, loc=loc)
            service_loc = (dep * (1 - proba_empty)).sum(axis=0) / (np.sum(dep, axis=0) + 0.001)
            service_ret = (arr * (1 - proba_full)).sum(axis=0) / (np.sum(arr, axis=0) + 0.001)
            service = 2 * mini(self.dep * service_loc, self.arr * service_ret)
        return service

    def compute_proba_matrix(self, distrib='NB'):
        """
        computes the probability of number of departure and arrivals matrix for a given distribution 
        :param distrib: the distribution of proba, default NB, available : Poisson (P) and Zero Inflated(ZI)
        :return: the matrix (timewindow*(2n_stations)*N)
        """
        self.N = 80
        mat = np.zeros((self.mean.shape[0], self.mean.shape[1], self.N))
        if distrib == 'NB':
            p = self.mean / self.var
            p = mini(p, 0.999) 
            r = maxi(1, (self.mean * p / (1 - p)))
            r = mini(150, r)
            n = np.array(list(map(int, r.flatten()))).reshape(r.shape)
            n += 1
            for k in range(self.N):
                mat[:, :, k] = nbinom.pmf(k, n=n, p=p)
                # a = comb(n + k - 1, k)
                # b = (p ** n) * ((1 - p) ** k)
                # mat[:, :, k] = a * b
        elif distrib == 'P':
            p = self.mean
            for k in range(self.N):
                mat[:, :, k] = poisson.pmf(k, p)
            sum = maxi(1 - mat.sum(axis=2), 0)
            mat[:, :, -1] += sum
        elif distrib == 'ZI':
            l = (self.var + self.mean ** 2) / self.mean - 1
            l[np.isnan(l)] = 0
            l[l < 0] = 0
            # l[self.mean == 0] = 0
            l = maxi(l, self.mean)
            # l = maxi(l, self.var)
            l = mini(l / self.mean, 50) * self.mean

            pi = 1 - self.mean / l
            # pi[l == 0] = 1
            # pi[l == 1000] = 1
            # pi[pi < 0] = 0
            for k in range(self.N):
                mat[:, :, k] = poisson.pmf(k, l)
                # np.exp(-l)*(l**k)/factorial(l)
                mat[:, :, k] *= 1 - pi
            mat[:, :, 0] += pi
            sum = maxi(1 - mat.sum(axis=2), 0)
            mat[:, :, -1] += sum

        # print('mat sum',(mat.sum(axis=2)-1).sum())
        assert (mat.sum(axis=2) - 1).sum() < 1e-6
        return mat

    def compute_service_level(self, distribution, current_capacity=None):
        """
        compute the service level, NB distribution supposed
        :param current_capacity: current network status, if None compute service for all statuses
        :return: service level : array if available_bikes set, else return a matrix with a service levels: each column 
            correspond to one station
        """
        if distribution == 'P':
            service = self.compute_service_level_Poisson(current_capacity)
        else:
            mat = self.compute_proba_matrix(distribution)
            service = self.compute_service_level_from_proba_matrix(mat, current_capacity)
        return service


if __name__ == '__main__':
    from model_station.ModelStations import ModelStations
    from preprocessing.Environment import Environment
    from preprocessing.Data import Data
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    ud = Environment('Bixi', 'train')
    d=Data(ud)
    mod= ModelStations(ud,'svd','gbt',**{'var':True})
    mod.train(d)
    mod.save()
    mod.load()
    i=43
    print(d.get_stations_col()[i], d.get_miniOD([])[d.get_stations_col()[i]].mean())
    sl = ServiceLevel(Environment('Bixi', 'train'), mod, 0.5)
    cmap = cm.binary
    s=0
    dd=[]
    for k in [10]:
        WT = mod.get_factors(d).iloc[:k,:]
        sl.compute_mean_var(WT,d,True)
        cap = sl.dict['capacities'][i]
        service = sl.compute_service_level_Poisson()
        np.savetxt('test.csv',service,)
        plt.plot(service[:,i][:cap], c=cmap(k/100 ))
        dd.append(np.log(((service[:,i]-s)**2).sum()))
        s=service[:,i]
        # print(service[5])
    plt.show()
    plt.plot(range(2,100,2),dd)
    plt.show()