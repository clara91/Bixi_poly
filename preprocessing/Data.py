from preprocessing.Station import *
from preprocessing.Environment import Environment
import config
from datetime import datetime, timedelta
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
        holliday = pd.read_csv(env.off_days, delimiter=',', quotechar='"')
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

        
        self.miniOD['total'] = pd.DataFrame(np.zeros(len(self.miniOD))) #TEM QUE REFAZER ESSA PARTE AQUI
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
    


    def get_miniOD_forecast(self,env, interval, length = [9, 2, 4, 4, 3, 2]):

        info = DB("info")
        #df =  pd.DataFrame(info.query_df('select * from weather_prediction'))
        df = info.query_df('select * from weather_prediction ORDER BY datetime desc limit 26')
        df = df[df.datetime >= datetime.now().replace(microsecond=0).replace(second=0).replace(minute=0)]
        holliday = pd.read_csv(env.off_days, delimiter=',', quotechar='"')
        holliday = pd.to_datetime(holliday).dt.date
        self.miniOD = pd.DataFrame(columns={'UTC timestamp','Annee','Date/Heure','Dir. du vent (10s deg)','Heure','Jour','Mois', 'Temps','temp','vent','wday','ferie','averses', 'neige',
            'pluie','fort','modere', 'verglas', 'bruine', 'poudrerie', 'brouillard', 'nuageux', 'orage', 'degage', 'precip', 'brr', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 
            'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'LV', 'MMJ', 'SD'})
        self.miniOD = self.miniOD[['UTC timestamp','Annee','Date/Heure','Dir. du vent (10s deg)','Heure','Jour','Mois', 'Temps','temp','vent','wday','ferie','averses', 'neige',
            'pluie','fort','modere', 'verglas', 'bruine', 'poudrerie', 'brouillard', 'nuageux', 'orage', 'degage', 'precip', 'brr', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 
            'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'LV', 'MMJ', 'SD']]
        #print(df['vent_direction'].iloc[-1])
        #pd.DataFrame(columns ={"dia","hora","wday"})
        #print(type(df['datetime']))
        
        for i in range(length[interval]):
            a=np.array([])
            ts = df['datetime'].iloc[-2-i]
            #ts = ts.replace(hour =i+1+int(df['datetime'].hour))
            
            # if ts.hour== 23:
            #     ts = ts +timedelta(hour=1)
            #print(ts)
            #print(type(ts))
            a = np.append(a,calendar.timegm(datetime.strptime(str(ts)[:13], env.precipitation_date_format).timetuple())) #'UTC timestamp'
            a = np.append(a,ts.year) #Annee
            a = np.append(a,ts) #Date/Heure
            a = np.append(a,vent_dire(df['vent_direction'].iloc[-1]))
            a = np.append(a,ts.hour) #Heure
            a = np.append(a,ts.day) #Jour
            a = np.append(a,ts.month) #Mois 
            a = np.append(a,df['cond_meteo_min'+str(i+1)].iloc[-2-i]) #Temps
            a = np.append(a,df['temp_min'+str(i+1)].iloc[-2-i]) #temp
            a = np.append(a,df['vent_kmh'].iloc[-1]) #vent
            a = np.append(a,ts.weekday()) #wday
            a = np.append(a,ts == holliday.any()) #ferie
            #print(type(df['cond_meteo_min1']))
            #'averses', 'neige','pluie','fort','modere', 'verglas', 'bruine', 'poudrerie', 'brouillard', 'nuageux', 'orage', 'degage'
            champs = env.weather_fields
            for ch in champs.keys():
                def f(x):
                    r = 0
                    for i in champs[ch]:
                        r += int(str(x).lower().find(i) != -1)
                    return r
                a = np.append(a,f(df['cond_meteo_min'+str(i+1)].iloc[-2-i]))
            #print(a[-1])
            a = np.append(a,a[14] | a[13] | a[12]|a[22]) #precip #r['pluie'] | r['neige'] | r['averses'] | r['orage']
            a = np.append(a,a[18]|a[20]) #brr #r['brouillard'] | r['bruine']
            #h
            for h in range(24):
                a = np.append(a, ts.hour == h)

            a = np.append(a,(ts.weekday() == 0) | (ts.weekday() == 4)) #LV
            a = np.append(a,(ts.weekday() == 1) | (ts.weekday() == 2) | (ts.weekday() == 3)) #MMJ
            a = np.append(a,(ts.weekday() == 5) | (ts.weekday() == 6)) #SD
            #print(a)
            self.miniOD.loc[i] = a
            #print(self.miniOD.loc[i])
            #print(self.miniOD)
        self.miniOD.to_csv("data_updated_forecast1.csv")
        return self.miniOD


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
        if start < 0:
            start = dim + start
        d.min = self.min + start / dim
        if n != -1:
            d.max = self.min + (start + n) / dim
        return d

def vent_dire(x):
        if(x =='N'):
            return 36
        elif(x == 'NW'):
            return 31.5
        elif(x == 'W'):
            return 27
        elif(x == 'SW'):
            return 22.5
        elif(x == 'S'):
            return 18
        elif(x == 'SE'):
            return 13.5
        elif(x == 'E'):
            return 9
        elif(x == 'NE'):
            return 4.5
        elif(x =='VR'):
            return -1
        else:
            return None

if __name__ == '__main__':
    from preprocessing.Environment import Environment

    ud = Environment('Bixi', 'train')
    d = Data(ud)
    #test1 = d.get_miniOD(log=False)
    #print(list(test1))
    test = d.get_miniOD_forecast(d.env,0)

    # print("informações sobre miniOD_database")
    # print(list(test))
    # print(len(list(test)))
    # # print(test.head())
    # print("informações sobre miniOD")
    # print(list(test1))
    # print(len(list(test1)))
    # 