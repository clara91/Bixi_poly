import os

import config
from preprocessing.Station import Stations
from preprocessing.Environment import Environment

# from preprocessing.UnprocessedSyntheticData import UnprocessedSyntheticData

os.chdir(config.root_path)
import pandas as pd
import numpy as np
import calendar
from datetime import datetime


##########################################
##                 utils                ##
##########################################
def data(name, unprocesseddata):
    def f3(x):
        if not (pd.isnull(x)):
            s = str(x)[:13]
            if ':' in s:
                i = s.index(':')
                s = s[:i]
            return calendar.timegm(datetime.strptime(s, unprocesseddata.get_dateformat()).timetuple())

        else:
            return 0

    path = unprocesseddata.trip_path
    if (not name.__contains__('2017')) or path.__contains__('synthetic'):
        r = pd.read_csv(open(path + name), delimiter=unprocesseddata.get_delimiter(), quotechar='"',
                        low_memory=False)
    else:
        r = pd.read_csv(open(path + name), delimiter=unprocesseddata.get_delimiter(), quotechar='"',
                        low_memory=False)  # [['end_date', 'start_date', 'start_station_code', 'end_station_code']]
        r = r[unprocesseddata.get_cols()]
        r.columns = [config.arr_prefix, config.dep_prefix, config.start_station_code, config.end_station_code]
    if unprocesseddata.system == 'capitalBS':
        if '2016' in name:
            r = r[unprocesseddata.get_cols() + ['Start station number', 'End station number']]
            s = Stations(unprocesseddata).s[['short_name', 'code']]
            r = pd.merge(r, s, how='left', left_on='Start station number', right_on='short_name')
            print('------------------------------------------')
            for i in np.unique(r['Start station number'][r['code'].isnull()]): print(i)
            r.drop('Start station', axis=1, inplace=True)
            r.drop('Start station number', axis=1, inplace=True)
            r.drop('short_name', axis=1, inplace=True)
            r.rename(columns={'code': 'Start station'}, inplace=True)
            r = pd.merge(r, s, how='left', left_on='End station number', right_on='short_name')
            r.drop('End station', axis=1, inplace=True)
            r.drop('End station number', axis=1, inplace=True)
            r.drop('short_name', axis=1, inplace=True)
            r.rename(columns={'code': 'End station'}, inplace=True)
        else:
            r = r[unprocesseddata.get_cols()]
            s = Stations(unprocesseddata).s[['name', 'code']]
            r = pd.merge(r, s, how='left', left_on='Start station', right_on='name')
            print('------------------------------------------')
            for i in np.unique(r['Start station'][r['code'].isnull()]): print(i)
            r.drop('Start station', axis=1, inplace=True)
            r.drop('name', axis=1, inplace=True)
            r.rename(columns={'code': 'Start station'}, inplace=True)
            r = pd.merge(r, s, how='left', left_on='End station', right_on='name')
            r.drop('End station', axis=1, inplace=True)
            r.drop('name', axis=1, inplace=True)
            r.rename(columns={'code': 'End station'}, inplace=True)
    return f3, r


def wd(x):
    """
    get x's week day
    :param x: timestamp
    :return: weekday

    """
    if not (pd.isnull(x)):
        return datetime.utcfromtimestamp(x).weekday()
    else:
        return 0


def complete(env, df, s=True):
    """
    fill the gaps in the dataframe
    :param env: environment
    :param df: dataframe
    :param s: true if trip are considered by station
    :return: the completed dataframe
    """
    min_t = int(min(df['UTC timestamp']))
    max_t = int(max(df['UTC timestamp']))
    n_h = (max_t - min_t) // 3600
    stations = Stations(env, None).get_ids()
    n_stations = len(stations)
    dfhs = np.zeros((n_h * n_stations, 2), dtype='int')
    dfhs[:, 0] = [i for i in range(min_t, max_t, 3600) for _ in range(n_stations)]
    dfhs[:, 1] = np.array(list(stations) * n_h)
    dfhs = pd.DataFrame(dfhs, columns=['UTC timestamp', 'station'])
    merged = pd.merge(dfhs, df, 'left', on=['UTC timestamp', 'station'])
    merged.fillna(value=0, inplace=True)
    # merged.iloc[:, 2][merged.iloc[:, 2].isnull()] = 0
    return merged


def load(name, env):
    path = env.precipitation_path
    df = pd.read_csv(path + name, delimiter=',', quotechar='"')
    # df.rename(index=str, columns=m, inplace=True)
    df['UTC timestamp'] = df['Date/Heure'].apply(
        lambda x: calendar.timegm(
            datetime.strptime(str(x)[:13], env.precipitation_date_format).timetuple()))
    df.loc[df['Temps'] == 'ND', 'Temps'] = np.nan
    li = ['Temps', 'vent', 'temp', 'visi',
          'pression', 'Hum']
    df[li].fillna(method='ffill')
    df[li].fillna(method='bfill')
    return df

 
##########################################
##       preprocess data and save       ##
##########################################
def compute_features(env, save=True):
    """
    computes the feature dataframe
    :param env: environement
    :param save: if true saves the result to unprocess_data.pre_per_hour_path
    :return: the dataframe
    """
    weather_files = env.get_meteo_files()
    frames = []
    for y in weather_files.keys():
        for file in weather_files[y]:
            try:
                frames.append(load(file, env))
            except FileNotFoundError:
                print('file not found : ' + file)
                pass
    r = pd.concat(frames)
    ##########################################
    ##       cast string to numbers         ##
    ##########################################
    for c in ['temp', 'visi', 'pression']:
        r[c] = r[c].apply(lambda x: float(str(x).replace(',', '.')))
    r.loc[r['pression'] == 0, 'pression'] = np.nan

    r.loc[r['pression'].isnull(), 'pression'] = np.nanmean(r['pression'].to_numpy())
    ##########################################
    ##       compute time features          ##
    ##########################################
    r['Annee'] = r['UTC timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).year)
    r['Mois'] = r['UTC timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).month)
    r['Jour'] = r['UTC timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).day)
    r['wday'] = r['UTC timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).weekday())
    r['Heure'] = r['UTC timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).hour)
    ##########################################
    ##    chargement des jours feries       ##
    ##########################################
    ferie = pd.read_csv(env.off_days, delimiter=',', quotechar='"')
    ferie['ferie'] = np.ones(ferie.shape[0], dtype=int)
    ferie['daycode'] = 380 * (ferie['annee'] - 2000) + ferie['mois'] * 31 + ferie['jour']
    ferie.drop(['jour', 'annee', 'mois'], inplace=True, axis=1)
    r['daycode'] = 380 * (r['Annee'] - 2000) + r['Mois'] * 31 + r['Jour']
    r = pd.merge(r, ferie, how='left', on='daycode')
    r.drop('daycode', inplace=True, axis=1)
    r.loc[r['ferie'].isnull(), 'ferie'] = 0
    ####################################################
    ##  transformation du temps n colonnes binaires   ##
    ####################################################
    #print("4")
    champs = env.weather_fields
    for ch in champs.keys():
        def f(x):
            r = 0
            for i in champs[ch]:
                r += int(str(x).lower().find(i) != -1)
            return r

        r[ch] = r['Temps'].apply(f)
    print("rrrrrr44444444444444")
    print(list(r))
    r['precip'] = r['pluie'] | r['neige'] | r['averses'] | r['orage']
    r['brr'] = r['brouillard'] | r['bruine']
    for h in range(24):
        r['h' + str(h)] = (r['Heure'] == h)
    r['LV'] = (r['wday'] == 0) | (r['wday'] == 4)
    r['MMJ'] = (r['wday'] == 1) | (r['wday'] == 2) | (r['wday'] == 3)
    r['SD'] = (r['wday'] == 5) | (r['wday'] == 6)
    print("rrrrrr55555555555555555")
    print(list(r))
    if save:
        #print("########################################33aqui")
        #print(env.pre_per_hour_path)

        r.to_pickle(env.pre_per_hour_path)

    return r


def compute_data_per_hour_per_station(env, save=True, year_stations=False):
    """
    creates the matrix with 4 columns : the hour, the station, the number of arrivals and the number of departure
    saves the result
    :param env: environment
    :return: dataframe
    """
    files = env.get_trip_files()
    framesstart = []
    framesend = []
    for y in files.keys():
        fstart = []
        fend = []
        for m in files[y]:
            f3, dstart = data(m, env)
            dend = dstart.copy()
            dstart['UTC timestamp'] = dstart[config.dep_prefix].apply(f3)
            dend['UTC timestamp'] = dend[config.arr_prefix].apply(f3)
            fstart.append(dstart.groupby(['UTC timestamp', config.start_station_code]).count()[[config.dep_prefix]])
            fend.append(dend.groupby(['UTC timestamp', config.end_station_code]).count()[[config.arr_prefix]])
            print(m, fstart[-1][config.dep_prefix].sum(), fend[-1][config.arr_prefix].sum())
        fstart = pd.concat(fstart)
        fend = pd.concat(fend)
        fstart['UTC timestamp'] = list(map(lambda x: x[0], fstart.index.values))
        fend['UTC timestamp'] = list(map(lambda x: x[0], fend.index.values))
        fstart['station'] = list(map(lambda x: x[1], fstart.index.values))
        fend['station'] = list(map(lambda x: x[1], fend.index.values))
        fstart.index = list(range(fstart.shape[0]))
        fend.index = list(range(fend.shape[0]))
        g = fend.groupby(['UTC timestamp', 'station']).sum()
        g['UTC timestamp'] = list(map(lambda x: x[0], g.index.values))
        g['station'] = list(map(lambda x: x[1], g.index.values))
        fend = g
        fend.index = fend.index.rename(['UTC index', 'station index'])
        fend = complete(env, fend)
        fstart = complete(env, fstart)
        print(str(y) + ' loaded', fstart.sum(), fend.sum())
        framesstart.append(fstart)
        framesend.append(fend)
    rstart = pd.concat(framesstart)
    rend = pd.concat(framesend)
    # rstart = rstart.groupby(['UTC timestamp', 'station']).sum()
    # rstart['UTC timestamp'] = list(map(lambda x: x[0], rstart.index.values))
    # rstart['station'] = list(map(lambda x: x[1], rstart.index.values))
    # rend = rend.groupby(['UTC timestamp', 'station']).sum()
    # rend['UTC timestamp'] = list(map(lambda x: x[0], rend.index.values))
    # rend['station'] = list(map(lambda x: x[1], rend.index.values))
    if save:
        rstart.to_pickle(env.data_dep_per_hour_per_station_path)
        rend.to_pickle(env.data_arr_per_hour_per_station_path)
    return rstart, rend


def combine_merged_s(env, save=True, precipitation=None, tripdeparture=None, triparrivals=None):
    """
    merges precipitations and trips one columns is created per hour.
    does not compute precipitation and trips matrix, only load them
    :param env:
    :return:
    """
    # fields to keep
    if precipitation is None:
        prec_per_h = env.load(env.pre_per_hour_path)
    else:
        prec_per_h = precipitation
    if tripdeparture is None: 
        trip_dep_h = env.load(env.data_dep_per_hour_per_station_path)
    else:
        trip_dep_h = tripdeparture
    if triparrivals is None:
        trip_arr_h = env.load(env.data_arr_per_hour_per_station_path)
    else:
        trip_arr_h = triparrivals

    m = pd.merge(trip_arr_h[[config.arr_prefix, 'UTC timestamp', 'station']], trip_dep_h[[config.dep_prefix, 'UTC timestamp', 'station']],
                 how='outer', on=['UTC timestamp', 'station'])
    # approximation : 0 to all hours that has no data
    for ch in [config.dep_prefix, config.arr_prefix]:
        m[ch].fillna(0,inplace=True)
    # print('dep loaded')
    # m['wday'] = m['UTC timestamp'].apply(lambda x: time.localtime(x).tm_wday)
    merged = pd.merge(m, prec_per_h, how='left', on='UTC timestamp')

    # merged = merged[merged['ferie'].isnull().apply(lambda x: not (x))]
    print('should be all 0', merged.isnull().sum())

    stations = Stations(env, None).get_ids()
    df = merged[merged['station'] == int(stations[0])]
    df.drop([config.arr_prefix, config.dep_prefix, 'station'], inplace=True, axis=1)
    df = df.reset_index()
    k = (merged['station'] == stations[0]).sum()
    merged.sort_values(by=['station', 'UTC timestamp'], inplace=True)
    startdate = merged[config.dep_prefix].to_numpy().reshape((int(merged.shape[0] / k), k)).T
    enddate = merged[config.arr_prefix].to_numpy().reshape((int(merged.shape[0] / k), k)).T
    startcol = list(map(lambda x: config.dep_prefix + ' ' + str(x), stations))
    endcol = list(map(lambda x: config.arr_prefix + ' ' + str(x), stations))
    start = pd.DataFrame(startdate, columns=startcol)
    hour = merged[merged['station'] == stations[0]]['UTC timestamp']
    hour = hour.reset_index()
    start['UTC timestamp'] = hour['UTC timestamp']
    end = pd.DataFrame(enddate, columns=endcol)
    end['UTC timestamp'] = hour['UTC timestamp']
    df = pd.merge(df, start, how='left', on='UTC timestamp')
    df = pd.merge(df, end, how='left', on='UTC timestamp')
    del start, end
    c_station=[]
    for s in stations:
        c_station.append(config.dep_prefix + ' ' + str(s))
        c_station.append(config.arr_prefix + ' ' + str(s))

    df['total']=df[c_station].to_numpy().sum(axis=1)

    c = [i for i in df.columns.values if not config.station_prefix in i]
    c+=c_station
    if not 'UTC timestamp' in c:
        c+=['UTC timestamp']
    df = df[c]
    df.fillna(0, inplace=True)
    df.sort_values(by=['Annee', 'Mois', 'Jour', 'Heure'], axis=0, inplace=True)
    if save:
        pd.to_pickle(df, env.station_df_satisfied_path)
    return df


def combine_lost_satisfied_demand(env, comb=True, save=True, station_df=None):
    """
    combine satisfied demand with estimated unsatisfied demand
    :param env: environment
    :param comb: if false only satisfied demand is kept
    :return: resulting dataframe (saves result)
    """
    if station_df is None:
        df_s = env.load(env.station_df_satisfied_path)
    else:
        df_s = station_df
    if comb:
        df_l = env.load(env.station_df_lost_path)
        df_s[df_l.columns] = df_s[df_l.columns].to_numpy() + df_l.to_numpy()
    if save:
        pd.to_pickle(df_s, env.station_df_path)
    return df_s


def get_features(env):
    """
        computes the feature dataframe
        :param env: environement
        :param save: if true saves the result to unprocess_data.pre_per_hour_path
        :return: the dataframe
        """
    weather_file = env.prevision_meteo

    frames = []
    file = weather_file
    try:
        df = pd.read_csv(open(file), delimiter=',', quotechar='"')
        # df.rename(index=str, columns=m, inplace=True)
        df['UTC timestamp'] = df['Date/Heure'].apply(
            lambda x: calendar.timegm(
                datetime.strptime(str(x)[:13], env.precipitation_date_format).timetuple()))
        df.loc[df['Temps'] == 'ND', 'Temps'] = np.nan
        li = ['Temps', 'vent', 'temp', 'visi',
              'pression', 'Hum']
        df[li].fillna(method='ffill')
        df[li].fillna(method='bfill')
        df['Temps'][df['Temps'].isnull()] = ''
        li.remove('Temps')
        df.fillna(value=0, inplace=True)
        # for i in li:
        #     df[i][df[i].isnull()] = 0
        frames.append(df)
    except FileNotFoundError:
        print('file not found : ' + file)
        pass
    r = pd.concat(frames)
    ##########################################
    ##       cast string to numbers         ##
    ##########################################

    for c in ['temp', 'visi', 'pression']:
        r[c] = r[c].apply(lambda x: float(str(x).replace(',', '.')))
    r.loc[r['pression'] == 0, 'pression'] = np.nan

    # r.loc[r['pression'].isnull(), 'pression'] = np.nanmean(r['pression'].to_numpy())
    r['pression'].fillna(np.nanmean(r['pression'].to_numpy()),inplace=True)
    r['pression'].fillna(100,inplace=True)
    # if r['pression'].isnull().sum() != 0:
    #     r['pression'][r['pression'].isnull()] = 100
    ##########################################
    ##       compute time features          ##
    ##########################################
    r['Annee'] = r['UTC timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).year)
    r['Mois'] = r['UTC timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).month)
    r['Jour'] = r['UTC timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).day)
    r['wday'] = r['UTC timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).weekday())
    r['Heure'] = r['UTC timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).hour)
    ##########################################
    ##    chargement des jours feries       ##
    ##########################################
    ferie = pd.read_csv(env.off_days, delimiter=',', quotechar='"')
    ferie['ferie'] = np.ones(ferie.shape[0], dtype=int)
    ferie['daycode'] = 380 * (ferie['Annee'] - 2000) + ferie['mois'] * 31 + ferie['jour']
    ferie.drop(['jour', 'Annee', 'mois'], inplace=True, axis=1)
    r['daycode'] = 380 * (r['Annee'] - 2000) + r['Mois'] * 31 + r['Jour']
    r = pd.merge(r, ferie, how='left', on='daycode')
    r.drop('daycode', inplace=True, axis=1)
    r.loc[r['ferie'].isnull(), 'ferie'] = 0
    ####################################################
    ##  transformation du temps n colonnes binaires   ##
    ####################################################
    champs = env.weather_fields
    for ch in champs.keys():
        def f(x):
            r = 0
            for i in champs[ch]:
                r += int(str(x).lower().find(i) != -1)
            return r

        r[ch] = r['Temps'].apply(f)
    r['precip'] = r['pluie'] | r['neige'] | r['averses'] | r['orage']
    r['brr'] = r['brouillard'] | r['bruine']
    for h in range(24):
        r['h' + str(h)] = (r['Heure'] == h)
    r['LV'] = (r['wday'] == 0) | (r['wday'] == 4)
    r['MMJ'] = (r['wday'] == 1) | (r['wday'] == 2) | (r['wday'] == 3)
    r['SD'] = (r['wday'] == 5) | (r['wday'] == 6)
    return r


def recompute_all_files(system, name, save=True):
    """
    method to launch all other methods
    :param system: environment name
    :param name: train or test set
    :return: None (all saved)
    """
    if save:
        env = Environment(system, name)
        # compute_features(env)
        # print('data per station')
        compute_data_per_hour_per_station(env) 
        # print('merge station')
        combine_merged_s(env)
        return combine_lost_satisfied_demand(env,comb=False) 
    else:
        env = Environment(system, name)
        precip = compute_features(env, save=True)
        # print('data per station')
        # dep, arr = compute_data_per_hour_per_station(env, save=True)
        # print('merge station')
        # merged = combine_merged_s(env, save=True, precipitation=precip, tripdeparture=dep,
        #                           triparrivals=arr)
        # return merged

if __name__ == '__main__':
    recompute_all_files('Bixi','train',False)
    recompute_all_files('Bixi','test',False)
    #recompute_all_files('citibike','train')
    #recompute_all_files('citibike','test')
    # recompute_all_files('capitalBS','train')
    # recompute_all_files('capitalBS','test')