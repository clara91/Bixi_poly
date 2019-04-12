from preprocessing.Station import *
from preprocessing.Environment import Environment
import config
from datetime import datetime
from bixi_pkg.db import DB
import calendar 

class Data(object):
    def __init__(self, env=None, first_name='Bixi', second_name=None):
        if env:
            self.env = env
        else:
            self.env = Environment(first_name, second_name)
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

    def get_OD(self):  # get and set
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
        return self.get_col('Start date ', since)

    def get_arr_cols(self, since):
        return self.get_col('End date ', since)

    def get_col(self, s, since):
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
        col = np.intersect1d(miniOD.columns.values, self.meteo)

        #miniOD = miniOD.sort_values(by='pdy timestamp')
        miniOD = miniOD.sort_values(by='UTC timestamp')
        for h in hours:
            for c in col:
                miniOD[c + str(h)] = miniOD[c].shift(periods=h)
        if hours != []:
            miniOD = miniOD[np.max(hours):]
        return miniOD

    def get_miniOD(self, hours=[], from_year=None, log=False, mean=False):
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
            res[self.get_stations_col(since=from_year)] = np.log(
                1 + res[self.get_stations_col(since=from_year)])
        if mean:
            m = res[self.get_stations_col(since=from_year)].mean(axis=0) + 0.01
            res[self.get_stations_col(
                since=from_year)] = res[self.get_stations_col(since=from_year)] / m
            return res, m

        return res
    def get_miniOD_database(self,env,hours=[],from_year=None, log=False, mean=False):
        info = DB("info")
        df =  pd.DataFrame(info.query_df('select * from weather_actuel'))
        holliday = pd.read_csv(env.off_days_update, delimiter=',', quotechar='"')
        holliday = pd.to_datetime(holliday).dt.date
        self.miniOD = pd.DataFrame()
        ###########################################################
        li = ['wind_speed', 'temperature_celsius', 'visibility',
              'pressure_kpa', 'humidity_pct']
        df[li].fillna(method='ffill')
        df[li].fillna(method='bfill')
        df.fillna(value=0, inplace=True)
        ###########################################################
        champs = env.weather_fields
        for ch in champs.keys():
            def f(x):
                r = 0
                for i in champs[ch]:
                    r += int(str(x).lower().find(i) != -1)
                return r
            self.miniOD[ch] = df['weather_condition'].apply(f)
        ###########################################################
        self.miniOD['UTC timestamp'] = df['date'].apply(
        lambda x: calendar.timegm(
            datetime.strptime(str(x)[:13], env.precipitation_date_format).timetuple()))
        ###########################################################
        for c in ['temperature_celsius', 'visibility', 'pressure_kpa']:
            df[c] = df[c].apply(lambda x: float(str(x).replace(',', '.')))
        df.loc[df['pressure_kpa'] == 0, 'pressure_kpa'] = np.nan

        # r.loc[r['pression'].isnull(), 'pression'] = np.nanmean(r['pression'].to_numpy())
        df['pressure_kpa'].fillna(np.nanmean(df['pressure_kpa'].to_numpy()),inplace=True)
        df['pressure_kpa'].fillna(100,inplace=True)
        ###########################################################
        self.miniOD['Annee'] = df['date'].dt.year
        self.miniOD['Mois'] = df['date'].dt.month
        self.miniOD['Jour']  =  df['date'].dt.day
        self.miniOD['Heure'] = df['date'].dt.hour
        self.miniOD['temp'] = df['temperature_celsius']
        self.miniOD['Hum'] = df['humidity_pct']
        self.miniOD['vent'] = df['wind_speed']
        self.miniOD['visi'] = df['visibility']
        self.miniOD['pression'] = df['pressure_kpa']
        self.miniOD['wday'] = df['date'].dt.weekday
        self.miniOD['ferie'] =  df['date'].dt.date.apply(lambda x: x == holliday.any())
        self.miniOD['precip'] = self.miniOD['pluie'] | self.miniOD['neige'] | self.miniOD['averses'] | self.miniOD['orage']
        self.miniOD['LV'] = (self.miniOD['wday'] == 0) | (self.miniOD['wday'] == 4)
        self.miniOD['MMJ'] = (self.miniOD['wday'] == 1) | (self.miniOD['wday'] == 2) | (self.miniOD['wday'] == 3)
        self.miniOD['SD'] = (self.miniOD['wday'] == 5) | (self.miniOD['wday'] == 6)
        ########################################################################
        for h in range(24):
            self.miniOD['h' + str(h)] = (self.miniOD['Heure'] == h)
        ##################################################total################# VER ISSO AQUIIIIIIIII

        # prec_per_h = env.load(env.pre_per_hour_path)
        # trip_dep_h = env.load(env.data_dep_per_hour_per_station_path)
        # trip_arr_h = env.load(env.data_arr_per_hour_per_station_path)

        # m = pd.merge(trip_arr_h[[config.arr_prefix, 'UTC timestamp', 'station']], trip_dep_h[[config.dep_prefix, 'UTC timestamp', 'station']],
        #              how='outer', on=['UTC timestamp', 'station'])
        # # approximation : 0 to all hours that has no data
        # for ch in [config.dep_prefix, config.arr_prefix]:
        #     m[ch].fillna(0,inplace=True)
        # # print('dep loaded')
        # # m['wday'] = m['UTC timestamp'].apply(lambda x: time.localtime(x).tm_wday)
        # merged = pd.merge(m, prec_per_h, how='left', on='UTC timestamp')

        # stations = Stations(env, None).get_ids()
        # df = merged[merged['station'] == int(stations[0])]
        # df.drop([config.arr_prefix, config.dep_prefix, 'station'], inplace=True, axis=1)
        # df = df.reset_index()
        # k = (merged['station'] == stations[0]).sum()
        # merged.sort_values(by=['station', 'UTC timestamp'], inplace=True)
        # startdate = merged[config.dep_prefix].to_numpy().reshape((int(merged.shape[0] / k), k)).T
        # enddate = merged[config.arr_prefix].to_numpy().reshape((int(merged.shape[0] / k), k)).T
        # startcol = list(map(lambda x: config.dep_prefix + ' ' + str(x), stations))
        # endcol = list(map(lambda x: config.arr_prefix + ' ' + str(x), stations))
        # start = pd.DataFrame(startdate, columns=startcol)
        # hour = merged[merged['station'] == stations[0]]['UTC timestamp']
        # hour = hour.reset_index()
        # start['UTC timestamp'] = hour['UTC timestamp']
        # end = pd.DataFrame(enddate, columns=endcol)
        # end['UTC timestamp'] = hour['UTC timestamp']
        # df = pd.merge(df, start, how='left', on='UTC timestamp')
        # df = pd.merge(df, end, how='left', on='UTC timestamp')
        # del start, end
        # c_station=[]
        # for s in stations:
        #     c_station.append(config.dep_prefix + ' ' + str(s))
        #     c_station.append(config.arr_prefix + ' ' + str(s))

        # self.miniOD['total']=df[c_station].to_numpy().sum(axis=1)
        self.miniOD['total'] = pd.DataFrame(np.zeros(len(self.miniOD))) #TEM QUE REFAZER ESSA PARTE AQUI
        ########################################################################
        # li = ['vent', 'temp', 'visi',
        #   'pression', 'Hum']
        # self.miniOD[li].fillna(method='ffill')
        # self.miniOD[li].fillna(method='bfill')
        # self.miniOD.fillna(value=0, inplace=True)
        
        # for c in ['temp', 'visi', 'pression']:
        #     self.miniOD[c] = self.miniOD[c].apply(lambda x: float(str(x).replace(',', '.')))
        # self.miniOD.loc[self.miniOD['pression'] == 0, 'pression'] = np.nan

        # self.miniOD.loc[self.miniOD['pression'].isnull(), 'pression'] = np.nanmean(self.miniOD['pression'].to_numpy())
        # self.miniOD.loc['pression'].fillna(100,inplace=True)
        #print(self.miniOD.head(100))
        ########################################################################
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
            res[self.get_stations_col(since=from_year)] = np.log(
                1 + res[self.get_stations_col(since=from_year)])
        if mean:
            m = res[self.get_stations_col(since=from_year)].mean(axis=0) + 0.01
            res[self.get_stations_col(
                since=from_year)] = res[self.get_stations_col(since=from_year)] / m
            return res, m
        return res
        ####### TEM QUE ADD ESSA PARTE AQUI
        # for c in ['temp', 'visi', 'pression']:
        #     r[c] = r[c].apply(lambda x: float(str(x).replace(',', '.')))
        # r.loc[r['pression'] == 0, 'pression'] = np.nan

        # r.loc[r['pression'].isnull(), 'pression'] = np.nanmean(r['pression'].to_numpy())
        # print(self.miniOD.head(100))
        # li = ['Temps', 'vent', 'temp', 'visi',
        #   'pression', 'Hum']
        # df[li].fillna(method='ffill')
        # df[li].fillna(method='bfill')
    

    def get_satisfied_demand(self):
        return self.env.load(self.env.station_df_satisfied_path)

    def get_synthetic_miniOD(self, hours, log, from_year=None):
        self.sminiOD = self.get_miniOD(hours, from_year, log)
        self.sminiOD['hh'] = self.sminiOD['UTC timestamp'].apply(
            lambda x: x % (3600 * 24 * 7))
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
        if start < 0:
            start = dim + start
        d.min = self.min + start / dim
        if n != -1:
            d.max = self.min + (start + n) / dim
        return d


if __name__ == '__main__':
    from preprocessing.Environment import Environment

    ud = Environment('Bixi', 'train')
    d = Data(ud)
    test1 = d.get_miniOD(log=False)
    test = d.get_miniOD_database(d.env)

    # print("informações sobre miniOD_database")
    # print(list(test))
    # print(len(list(test)))
    # # print(test.head())
    print("informações sobre miniOD")
    # print(list(test1))
    # print(len(list(test1)))
    # 