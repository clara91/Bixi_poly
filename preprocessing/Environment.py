import os

import numpy as np
import pandas as pd

from config import data_path
import datetime

class Environment(object):
    def __init__(self, name, second_name):
        self.system = name
        if second_name[-1] != '/': second_name += '/'
        # name of storing directory
        self.name = second_name
        self.data_path = data_path + name + '/'
        #creating dirs
        if not os.path.exists(self.data_path + second_name):
            os.makedirs(self.data_path + second_name)
            os.makedirs(self.data_path+second_name+'temp/')
        elif not os.path.exists(self.data_path + second_name+'temp/'):
            os.makedirs(self.data_path + second_name + 'temp/')
        # directory path
        self.aggregated = self.data_path + second_name
        if not os.path.exists(self.aggregated):
            os.makedirs(self.aggregated)
        # path to station information
        self.station_info = self.data_path + 'hitorique_deplacements/station_info.csv'
        # path to meteo files
        self.precipitation_path = self.data_path + 'precipitations/'
        # path to empty/full events
        self.activity_path = self.data_path + 'etat_stations/'
        # path to trip history
        self.trip_path = self.data_path + 'hitorique_deplacements/'
        # path to first trip aggregation
        self.off_days = self.data_path + "type_de_jour/jours_feries.csv"
        self.off_days_update = self.data_path + "type_de_jour/holiday_days.csv"

        self.holidays = self.data_path + "type_de_jour/vacances.csv"
        self.trip_per_hour_start_path = self.aggregated + 'trip_per_hour_start.pdy'
        self.trip_per_hour_end_path = self.aggregated + 'trip_per_hour_end.pdy'
        # path to preprocessed meteo file
        self.pre_per_hour_path = self.aggregated + 'pre_per_hour.pdy'
        # path to 2nd trip aggregation
        self.data_dep_per_hour_per_station_path = self.aggregated + 'data_dep_per_hour_per_station.pdy'
        self.data_arr_per_hour_per_station_path = self.aggregated + 'data_arr_per_hour_per_station.pdy'
        # path to OD matrix
        self.OD_path = self.aggregated + 'data_OD.npy'
        # path to OD matrix per hour
        self.OD_per_hour_path = self.aggregated + 'data_OD_hour.pdy'
        # path to trip and precipitation merged file
        self.merged_h_path = self.aggregated + 'merged_h.pdy'
        # path to fully preprocessed file of historical trips
        self.station_df_satisfied_path = self.aggregated + 'station_satisfied_df.pdy'
        self.temppath = self.aggregated + 'temp/'
        # path of estimation of lost trips
        self.station_df_lost_path = self.aggregated + 'station_lost_df.pdy'
        # path to fully aggregated dataframe
        self.station_df_path = self.aggregated + 'station_df.pdy'
        # path to fully aggregated dataframe
        self.sumOD = self.aggregated + 'sumOD.pdy'
        # path to fully aggregated dataframe with decorrelated features
        self.station_decor_path = self.aggregated + 'station_df_decor.pdy'

        self.decision_intervals = self.aggregated + 'decision_interval_min_max_target.csv'

        self.prevision_meteo = self.data_path + 'prevision.csv'
        self.precipitation_date_format = '%Y-%m-%d %H'
        self.weather_fields = {
            'averses': ['averses','rainshower showers'], #short period of rain
            'neige': ['neige','snow','snowshower'], #snow
            'pluie': ['pluie','rain'], #rain
            'fort': ['fort','mainly','heavy'], #strong,heacy
            'modere': ['modere', 'modã©rã©e','partly','mostly','light','moderate'], #moderate
            'verglas': ['vergla','freezing rain'], #black ice
            'bruine': ['bruine','drizzle'], #drizzle
            'poudrerie': ['poudrerie', 'granules', 'ice'], #ice
            'brouillard': ['brouillard', 'brume','mist','fog','smoke'], #fog
            'nuageux': ['nuageux','cloudy'], #cloudy
            'orage': ['orage','thunderstorms'], #thunderstorm
            'degage': ['degage', 'dã©gagã©','sunny','clear'],

        }

    def load(self, path):
        if path.__contains__('.pdy'):
            return pd.read_pickle(path)
        elif path.__contains__('.npy'):
            return np.load(path)
        elif '.csv' in path:
            return pd.read_csv(path)

    def get_trip_files(self):
        if self.system == 'Bixi':
            if self.name.__contains__('train'):
                return {
                    2015: ['OD_2015-04.csv', 'OD_2015-05.csv', 'OD_2015-06.csv', 'OD_2015-07.csv', 'OD_2015-08.csv',
                           'OD_2015-09.csv', 'OD_2015-10.csv', 'OD_2015-11.csv'],
                    2016: ['OD_2016-05.csv', 'OD_2016-06.csv', 'OD_2016-07.csv', 'OD_2016-08.csv', 'OD_2016-09.csv',
                           # 'OD_2016-10.csv', 'OD_2016-11.csv'
                           ],
                    # 2017: [
                    #     'OD_2017-04.csv', 'OD_2017-05.csv',
                    # ]
                }
            if self.name.__contains__('test'):
                return {
                    2016: ['OD_2016-10.csv', 'OD_2016-11.csv']
                    # 2017: [
                    #     'OD_2017-06.csv',
                    #     'OD_2017-07.csv',
                    # ]
                }
        if self.system == 'capitalBS':
            if self.name.__contains__('train'):
                return {
                    2015: ['2015-Q1-Trips-History-Data.csv', '2015-Q2-Trips-History-Data.csv',
                           '2015-Q3-Trips-History-Data.csv', '2015-Q4-Trips-History-Data.csv'],
                    2016: ['2016-Q1-Trips-History-Data.csv', '2016-Q2-Trips-History-Data.csv',
                           '2016-Q3-Trips-History-Data-1.csv', '2016-Q3-Trips-History-Data-2.csv'],
                }
            if self.name.__contains__('test'):
                return {
                    2016: ['2016-Q4-Trips-History-Data.csv']
                }
        if self.system == 'citibike':
            if self.name.__contains__('train'):
                return {
                    2015: ['2015' + str(x).zfill(2) + '-citibike-tripdata.csv' for x in range(1, 13)],
                    2016: ['2016' + str(x).zfill(2) + '-citibike-tripdata.csv' for x in range(1, 10)]
                }
            if self.name.__contains__('test'):
                return {
                    2016: ['2016' + str(x) + '-citibike-tripdata.csv' for x in [10,11,12]]
                }

    def get_activity_files(self):
        return {'years': self.get_trip_files().keys()}

    def get_meteo_files(self):
        n = ['31', '28', '31', '30', '31', '30', '31', '31', '30', '31', '30', '31']
        nb = ['31', '29', '31', '30', '31', '30', '31', '31', '30', '31', '30', '31']
        files = {
            2015: [],
            2016: [],
            2017: []
        }
        for y in list(files.keys()):
            if y % 4 != 0:
                for m in range(1, 13):
                    if m < 10:
                        files[y].append(
                            'fre-hourly-0' + str(m) + '01' + str(y) + '-0' + str(m) + n[m - 1] + str(y) + '.csv')
                    else:
                        files[y].append(
                            'fre-hourly-' + str(m) + '01' + str(y) + '-' + str(m) + n[m - 1] + str(y) + '.csv')
            else:
                for m in range(1, 13):
                    if m < 10:
                        files[y].append(
                            'fre-hourly-0' + str(m) + '01' + str(y) + '-0' + str(m) + nb[m - 1] + str(y) + '.csv')
                    else:
                        files[y].append(
                            'fre-hourly-' + str(m) + '01' + str(y) + '-' + str(m) + nb[m - 1] + str(y) + '.csv')

        return files

    def get_cols(self, rename=False):
        """
        name of desired columns or dictionary to convert them to standard name
        :param rename: if trus dict, else name
        :return: list of columns names
        """
        if not rename:
            if self.system == 'Bixi':
                return ['End date', 'Start date', 'End station', 'Start station']
            if self.system=='capitalBS':
               return ['Start date','End date', 'End station', 'Start station']
            if self.system == 'citibike':
                return ['stoptime', 'starttime', 'end station id', 'start station id']
        else:
            if self.system == 'Bixi':
                return {}
            if self.system == 'capitalBS':
                return {}
                # return {'End station number': 'End station',
                #         'Start station number': 'Start station'
                #         }
            if self.system == 'citibike':
                return {'stoptime': 'End date',
                        'starttime': 'Start date',
                        'end station id': 'End station',
                        'start station id': 'Start station'
                        }

    def get_delimiter(self):
        """
        delimiter of the trip data csv file
        :return: ',' or ';'
        """
        if self.system == 'Bixi':
            return ';'
        else:
            return ','

    def get_dateformat(self):
        if self.system == 'Bixi':
            return "%Y-%m-%d %H"
        if self.system == 'capitalBS':
            return "%m/%d/%Y %H"
        if self.system =='citibike':
            if self.name == 'test/':
                return "%Y-%m-%d %H"
            else:
                return "%m/%d/%Y %H"

    def build_dummy_prevision(self, year=None, month=None, day=None):
        files = self.get_meteo_files()
        cols = pd.read_csv(self.precipitation_path + files[2016][0]).columns.values
        #print(cols)
        if year is None or month is None or day is None:
            t = datetime.datetime.now()
        else:
            t = datetime.datetime(year, month, day)
        delta = datetime.timedelta(hours=1)
        a = np.zeros((24 * 10), dtype=type(t))
        for i in range(24 * 10):
            a[i] = t
            t = t + delta

        df = pd.DataFrame(np.zeros((24 * 10, len(cols)), dtype=int), columns=cols)
        df['Temps'] = "ND"
        df['Date/Heure'] = a
        df['Date/Heure'] = df['Date/Heure'].apply(
            lambda x: str(x.year) + '-' + str(x.month).zfill(2) + '-' + str(x.day).zfill(2) + ' ' + str(
                x.hour).zfill(2))
        df.to_csv(self.prevision_meteo, index=False)
        #print(df)

if __name__ == '__main__':
    ud = Environment('Bixi', 'release')
    ud.build_dummy_prevision(2017, 8, 24)