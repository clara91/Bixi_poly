from preprocessing.Station import *
from preprocessing.Environment import Environment
import config

class Data(object):
    def __init__(self, env=None, first_name='Bixi', second_name=None):
        if env:
            self.env = env
        else:
            self.env=Environment(first_name,second_name)
        self.stations = None  # stations numbers
        self.OD = None  # OD matrix (no temporal aspect)
        self.hour_OD = None  # OD matrix per hour
        self.miniOD = None  # arr and dep per station and per hour
        self.ODsum = None  # arr and dep per hour (whole network)
        self.cols = None  # names of stations columns
        self.decorminiOD = None  # decorelated(features) OD matrix
        self.min = 0
        self.max = 1
        self.meteo = [
            'temp',
            # 'vent',
            'precip',
            # 'visi',
            # 'averses',
            # 'pression',
            # 'fort',
            # 'nuageux',
            'brr',
            'Hum',
            # 'neige',
            'pluie',
            # 'verglas',
            # 'bruine',
            # 'poudrerie',
            # 'brouillard',
            # 'orage',
        ]

    def get_OD(self): #get and set
        if self.OD is None:
            self.OD = self.env.load(self.env.OD_path)
        return self.OD

    def get_stations_col(self, since=None):
        cols = []
        for s in self.get_stations_ids(since):
            cols.append(config.arr_prefix + ' ' + str(s))
            cols.append(config.dep_prefix + ' ' + str(s))
        return cols

    def get_dep_cols(self, since):
        return self.get_col('Start date ',since)

    def get_arr_cols(self, since):
        return self.get_col('End date ',since)

    def get_col(self,s,since):
        cols = []
        for st in self.get_stations_ids(since):
            cols.append(s + str(int(st)))
        return cols

    def get_hour_OD(self):
        return self.env.load(self.env.OD_per_hour_path)

    def get_stations_ids(self, since):
        if (self.stations is None) or (self.stations.since != since):
            self.stations = Stations(self.env, since)
            self.since = since
        return self.stations.get_ids()

    def get_stations_pks(self, since):
        if (self.stations is None) or (self.stations.since != since):
            self.stations = Stations(self.env, since)
            self.since = since
        return self.stations.get_pks()

    def get_stations_capacities(self, since, tab=None):
        if self.stations is None:
            self.stations = Stations(self.env, since)
        self.stations.since = since
        return self.stations.get_capacities(tab)

    def get_stations_loc(self, since=None):
        if self.stations is None:
            self.stations = Stations(self.env, since)
        return self.stations.get_loc()

    def towindow_features(self, miniOD, hours):
        #print(miniOD.columns.values)
        col = np.intersect1d(miniOD.columns.values, self.meteo) #Find the intersection of two arrays.

        #miniOD = miniOD.sort_values(by='pdy timestamp')
        miniOD = miniOD.sort_values(by='UTC timestamp')
        for h in hours:
            for c in col:
                miniOD[c + str(h)] = miniOD[c].shift(periods=h)
        if hours != []:
            miniOD=miniOD[np.max(hours):]
        return miniOD

    def get_miniOD(self, hours, from_year=None, log=False, mean=False):
        if self.miniOD is None:
            self.miniOD = self.env.load(self.env.station_df_path)
            r = self.miniOD.shape[0]
            self.miniOD = self.miniOD[int(self.min * r):int(self.max * r)]
        if from_year is None:
            res = self.miniOD
        elif from_year == 2015:
            res = self.miniOD[self.miniOD['Annee'] != 2017]
        elif from_year == 2017:
            res = self.miniOD[self.miniOD['Annee'] == 2017]
        else:
            res = self.miniOD[self.miniOD['Annee'] >= from_year]

        res = self.towindow_features(res, hours)

        if log:
            res[self.get_stations_col(since=from_year)] = np.log(1 + res[self.get_stations_col(since=from_year)])
        if mean:
            m = res[self.get_stations_col(since=from_year)].mean(axis=0)+0.01
            res[self.get_stations_col(since=from_year)] = res[self.get_stations_col(since=from_year)] / m
            return res, m

        return res

    def get_satisfied_demand(self):
        return self.env.load(self.env.station_df_satisfied_path)

    def get_synthetic_miniOD(self, hours, log, from_year=None):
        self.sminiOD = self.get_miniOD(hours, from_year, log)
        self.sminiOD['hh'] = self.sminiOD['UTC timestamp'].apply(lambda x: x % (3600 * 24 * 7))
        self.sminiOD = self.sminiOD.groupby('hh').mean()
        return self.sminiOD

    def get_ODsum(self):
        if self.ODsum is None:
            self.ODsum = self.env.load(self.env.sumOD)
        r = self.ODsum.shape[0]
        return self.ODsum[r * self.min:r * self.max]

    def get_partialdata_per(self, min, max):
        d = Data(self.env)
        l = self.max - self.min

        d.max -= (1 - max) * l
        d.min += min * l
        return d

    def get_partialdata_n(self, start, n):
        d = Data(self.env)
        dim = self.env.load(self.env.station_df_path).shape[0]
        if start<0:
            start=dim+start
        d.min=self.min+start/dim
        if n!=-1:
            d.max=self.min+(start+n)/dim
        return d


if __name__ == '__main__':
    from preprocessing.Environment import Environment

    ud = Environment('Bixi', 'train')
    d = Data(ud)
    d.get_miniOD(log=False)
