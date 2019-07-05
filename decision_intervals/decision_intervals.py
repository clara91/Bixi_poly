
from ServiceLevel import ServiceLevel
from code_v1.utils.modelUtils import *
from code_v1.model_station.ModelStations import ModelStations
from code_v1.preprocessing.Data import Data
from code_v1.preprocessing.Environment import Environment
import sys
import operator 

from datetime import datetime

class DecisionIntervals(object):
    """
    class for computing the decision intervals, uses the Service level class
    """
    def __init__(self, env, mod, arr_vs_dep, beta):
        """
        :param env: environment
        :param mod: model
        :param arr_vs_dep: alpha value (value of arrivals vs departures)
        :param beta: strength of intervals
        """
        #self.hours = [0, 6, 11, 15, 20]
        #self.length = [6, 5, 4, 5, 4]
        #self.hours = [0, 9, 11, 15, 19, 22]
        #self.length =  [9, 2, 4, 4, 3, 2]
        self.hours = [22, 6, 9, 11, 15.5, 19]
        # #self.length =  [8, 3, 2, 4.5, 3.5, 3]
        self.length =  [8, 3, 2, 5, 4, 3]
        self.SL = ServiceLevel(env, mod, arr_vs_dep)
        self.param_beta = beta

    def compute_decision_intervals(self, WT, data, predict=True, **kwargs):
        """
        computes decision intervals for WT (Weather and temporal features) data
        :param WT: the features for the next hours, the 
        :param data: the learning data (class Data)
        :param predict: if True use predictions, if False use real data
        :return: (min, target, max), each element being a array containing one value per station
        """
        hparam = {
            'distrib': 'P'
        }
        hparam.update(kwargs)
        # print(hparam)
        self.SL.compute_mean_var(WT, data, predict)
        service = self.SL.compute_service_level(hparam['distrib'])
        #print("2")
        best_inv = np.argmax(service, axis=0)
        best_serlev = service[best_inv, range(service.shape[1])]
        service[service == 0] = 2
        worst_inv = np.argmin(service, axis=0)
        worst_serlev = service[worst_inv, range(service.shape[1])]
        worst_serlev[worst_serlev == 2] = 0
        service[service == 2] = 0
        service_min_to_assure = worst_serlev + (best_serlev - worst_serlev) * self.param_beta
        b = service >= service_min_to_assure
        b2 = np.cumsum(b, axis=0)
        #print("3")
        min_inv = np.argmax(b, axis=0)
        max_inv = np.argmax(b2, axis=0)

        return min_inv, best_inv, max_inv

    def compute_min_max_data(self, WT, data, predict=True, save=True, **kwargs):
        """
        compute the min/max/target intervals starting from the first monday in WT and saves it to file 
        :param WT: features to predict on 
        :param data: the enviroment data (Data object) for station information
        :param predict: predict: if True use predictions, if False use real data
        :return: intervals data frame
        """
        tw = 10
        cond = True
        i0 = 0
        # while cond:
        #     print((WT['wday'].to_numpy()[i0]))
        #     print(WT['Heure'].to_numpy()[i0])
        #     cond = not ((WT['wday'].to_numpy()[i0] == 6) and (WT['Heure'].to_numpy()[i0] == 0)) #sunday midnight
        #     i0 += 1
        # i0 -= 1
        cols_min = []
        cols_max = []
        cols_target = []
        for i in range(1, 8):
            for t in range(1, 5):
                cols_min.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Min')
                cols_target.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Target')
                cols_max.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Max')
            cols_min.append('dayOfWeek' + str(i) + '.period' + str(5) + 'min')
            cols_target.append('dayOfWeek' + str(i) + '.period' + str(5) + 'target')
            cols_max.append('dayOfWeek' + str(i) + '.period' + str(5) + 'max')
        inter_min = pd.DataFrame(np.zeros((data.get_stations_capacities(None).shape[0], 7 * 5)),
                                 index=data.get_stations_ids(None), columns=cols_min)
        inter_max = pd.DataFrame(np.zeros((data.get_stations_capacities(None).shape[0], 7 * 5)),
                                 index=data.get_stations_ids(None), columns=cols_max)
        inter_target = pd.DataFrame(np.zeros((data.get_stations_capacities(None).shape[0], 7 * 5)),
                                    index=data.get_stations_ids(None), columns=cols_target)

        for d in range(7):
            for h in range(len(self.hours)): #[0, 9, 11, 15, 19, 22] #[6, 5, 4, 5, 4]
                print(i0 + 24 * d + self.hours[h] + tw, end='\r')
                assert (WT['wday'].iloc[i0 + 24 * d + self.hours[h]] == (d - 1) % 7)
                m, t, M = self.compute_decision_intervals(
                    WT.iloc[i0 + 24 * d + self.hours[h]:i0 + 24 * d + self.hours[h] + tw, :], data,
                    predict, **kwargs)
                m = m / data.get_stations_capacities(None)
                t = t / data.get_stations_capacities(None)
                M = M / data.get_stations_capacities(None)
                inter_min.to_numpy()[:, 5 * d + h] = m
                inter_target.to_numpy()[:, 5 * d + h] = t
                inter_max.to_numpy()[:, 5 * d + h] = M

        min_max = pd.read_csv(open(config.root_path + 'resultats/decision_interval_min_max_target_new.csv'), sep=',')
        min_max[cols_min] = np.nan
        min_max[cols_max] = np.nan
        min_max[cols_target] = np.nan
        index_min_max = min_max['info.terminalName']

        def f(x):
            try:
                int(x)
                return int(x)
            except ValueError:
                return -1

        index_min_max = list(map(f, index_min_max))
        min_max = min_max.set_index([index_min_max])
        min_max.update(inter_min * 100)
        min_max.update(inter_max * 100)
        min_max.update(inter_target * 100)
        if save:
            min_max.to_csv(data.env.decision_intervals[:-4] + kwargs['distrib'] + '.csv', index=False)
        return min_max

    def general_min_max(self, WT, data, predict=True, **kwargs):
        """
        min max function independent of the weather

        To be tested and completed and improved (very slow)

        :param WT: features
        :param data: environment data (Data object)
        :param predict: if True use predictions, if False use real data
        :param kwargs:
        :return: interval dataframe
        """
        hparam = {
            'distrib': 'NB'
        }
        hparam.update(kwargs)
        min_max = None
        k = 0
        n = int(WT.shape[0] / 7 / 24)
        n = 2
        print()
        for k in range(0, n - 1):
            i = k * 24 * 7
            print(i)
            m = self.compute_min_max_data(WT[i:], data, predict, save=False, **hparam)
            if min_max is None:
                min_max = m
            else:
                min_max += m
        min_max /= n
        min_max.to_csv(data.env.decision_intervals[:-4] + hparam['distrib'] + '.csv', index=False)
        print('end general int ' + hparam['distrib'])
        return min_max

    def transform_to_intervals(self, data, df):
        """
        read bixi intervals and reshape it
        :param data: environment
        :param df: bixi intervals dataframe
        :return: 3 dataframe containing the minimum, the maximum and the target for each station for each period
        """
        df['code'] = df['info.terminalName']

        def f(x):
            try:
                return int(x)
            except:
                return -1

        df.loc[:, 'code'] = df['code'].apply(f)
        df.set_index(df['code'], inplace=True)
        df.drop(-1, axis=0, inplace=True)
        df['info.capacity'].update(data.get_stations_capacities(None, df['code'].to_numpy()))
        rowsmin = {}
        rowsmax = {}
        rowstar = {}
        # renommer les lignes
        for i in range(1, 8):
            i_prime = ((i - 5) % 7)
            for t in range(1, 5):
                rowsmin['dayOfWeek' + str(i) + '.period' + str(t) + 'Min'] = 24 * i_prime + self.hours[t - 1]  # +' Min'
                rowstar['dayOfWeek' + str(i) + '.period' + str(t) + 'Target'] = 24 * i_prime + self.hours[t - 1]
                rowsmax['dayOfWeek' + str(i) + '.period' + str(t) + 'Max'] = 24 * i_prime + self.hours[t - 1]  # +' Max'
            rowsmin['dayOfWeek' + str(i) + '.period' + str(5) + 'min'] = 24 * i_prime + self.hours[4]  # +' Min'
            rowstar['dayOfWeek' + str(i) + '.period' + str(5) + 'target'] = 24 * i_prime + self.hours[4]
            rowsmax['dayOfWeek' + str(i) + '.period' + str(5) + 'max'] = 24 * i_prime + self.hours[4]  # +' Max'
        for val in [list(rowsmin.keys()), list(rowsmax.keys()), list(rowstar.keys())]:
            val.sort()
            df.loc[:, val] = mini(maxi(df.loc[:, val] / 100,0),1)
            for i in range(len(val)):
                df.loc[:, val[i]] = df.loc[:, val[i]] * df.loc[:, 'info.capacity']
            for _ in range(2):
                for i in range(len(val)):
                    df.loc[df[val[i]].isnull(), val[i]] = df[val[i - len(self.hours)]][df[val[i]].isnull()]
        dfmin = df[list(rowsmin.keys())]
        dfmax = df[list(rowsmax.keys())]
        dftar = df[list(rowstar.keys())]
        dfmin.rename(columns=rowsmin, inplace=True)
        dfmax.rename(columns=rowsmax, inplace=True)
        dftar.rename(columns=rowstar, inplace=True)
        dfmin = dfmin.transpose()
        dftar = dftar.transpose()
        dfmax = dfmax.transpose()
        rm = {}
        rM = {}
        rt = {}
        for col in dfmin.columns.values:
            rm[col] = 'Min ' + str(col)
            rM[col] = 'Max ' + str(col)
            rt[col] = 'Tar ' + str(col)
        dfmin.rename(columns=rm, inplace=True)
        dfmax.rename(columns=rM, inplace=True)
        dftar.rename(columns=rt, inplace=True)
        return dfmin, dfmax, dftar

    def load_intervals(self, data, path, distrib):
        """
        load decision intervals in a df hours, 3*nstations, 1 col max, min et target per station
        :param data: environment
        :return: df
        """
        path = path[:-4] + distrib + path[-4:]
        df = pd.read_csv(path, sep=',', encoding='latin-1')
        return self.transform_to_intervals(data, df)

    def compute_mean_intervals(self, data, path, distrib):
        """
        computes the average length of intervals
        :param data: environment
        :param path: interval file
        :param distrib: distribution hypothesis
        :return: average length of intervals
        """
        path = path[:-4] + distrib + path[-4:]
        df = pd.read_csv(path, sep=',', encoding='latin-1')
        df['code'] = df['info.terminalName']

        def f(x):
            try:
                return int(x)
            except:
                return -1

        df.loc[:, 'code'] = df['code'].apply(f)
        df.set_index(df['code'], inplace=True)
        df.drop(-1, axis=0, inplace=True)
        df['info.capacity'].update(data.get_stations_capacities(None, df['code'].to_numpy()))
        rowsmin = []
        rowsmax = []
        for i in range(1, 8):
            for t in range(1, 5):
                rowsmin.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Min')
                rowsmax.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Max')
            rowsmin.append('dayOfWeek' + str(i) + '.period' + str(5) + 'min')
            rowsmax.append('dayOfWeek' + str(i) + '.period' + str(5) + 'max')
        for val in [list(rowsmin), list(rowsmax)]:
            val.sort()
            df.loc[:, val] = df.loc[:, val] / 100
            for i in range(len(val)):
                df.loc[:, val[i]] = df.loc[:, val[i]] * df.loc[:, 'info.capacity']
            for _ in range(2):
                for i in range(len(val)):
                    df.loc[df[val[i]].isnull(), val[i]] = df[val[i - len(self.hours)]][df[val[i]].isnull()]

        m = df[rowsmin].to_numpy()
        M = df[rowsmax].to_numpy()
        return np.nanmean(M - m)

    def mean_alerts(self, test_data, intervals, df):
        """
        compute the number of rebalancing operations needed to keep the network balanced
        :param test_data: the environment of the test
        :param intervals: the intervals (3 dataframes) to test
        :param df: the data used to test the intervals
        :return: the average number of rebalancing per day dep/arr, and their repartition during the day
        """
        intermin, intermax, intertar = intervals
        intermin=intermin[test_data.get_col('Min ',None)].astype(np.double)
        intertar=intertar[test_data.get_col('Tar ',None)].astype(np.double)
        intermax=intermax[test_data.get_col('Max ',None)].astype(np.double)
        intermin.rename(lambda x:x[4:],axis=1,inplace=True)
        intermax.rename(lambda x:x[4:],axis=1,inplace=True)
        intertar.rename(lambda x:x[4:],axis=1,inplace=True)
        intermin = np.round(intermin)
        intertar = np.round(intertar)
        intermax = np.round(intermax)
        tar_min=intertar-intermin
        max_tar=intermax-intertar
        mat = df[test_data.get_arr_cols(None)].to_numpy() - df[test_data.get_dep_cols(None)].to_numpy()
        print(df['Heure'].to_numpy()[0])
        print(mat.sum())
        def f(h):
            if h < 6:
                return 0
            elif h < 11:
                return 6
            elif h < 15:
                return 11
            elif h < 20:
                return 15
            else:
                return 20
        features = df['wday'] * 24 + df['Heure'].apply(f)
        inv = intertar.loc[features[0], :].to_numpy()
        r1 = np.zeros((24,inv.shape[0]))
        r2 = np.zeros((24,inv.shape[0]))
        for i in range(mat.shape[0]):
            inv += mat[i, :]
            b1=np.array([1])
            b2=np.array([1])
            while b1.sum()>0 or b2.sum()>0:
                b1 = inv<intermin.loc[features[i],:].to_numpy()
                b2 = inv>intermax.loc[features[i],:].to_numpy()
                r1[i%24] += b1
                r2[i%24] += b2
                inv[b1] += maxi(tar_min.loc[features[i],:].to_numpy()[b1],1)
                inv[b2] -= maxi(max_tar.loc[features[i],:].to_numpy()[b2],1)
                inv = np.round(inv)
                # print(inv)
        r1s = r1.sum(axis=1)/(mat.shape[0])*24
        r2s = r2.sum(axis=1)/(mat.shape[0])*24
        return r1.sum()/(mat.shape[0])*24,r2.sum()/(mat.shape[0])*24,r1s,r2s

    def eval_worst_case(self, test_data, distrib, intervals, df, single=False, arr_dep=None):
        """
        computes the worst case test
        :param test_data: environment
        :param distrib: not used
        :param intervals: intervals dataframes (3 dataframes)
        :param df: the testing data (dataframe)
        :param single: to test only the first occurence of each period
        :param arr_dep: 'arr' or 'dep' to get worst case analysis for arrivals or departures
        :return: worst test case value
        """
        intermin, intermax, intertar = intervals  # DI.load_intervals(test_data, path, distrib)
        intermin.columns = list(map(lambda x: x.replace('Min', 'Start date'), intermin.columns.values))
        intermax.columns = list(map(lambda x: x.replace('Max', 'End date'), intermax.columns.values))

        # diff=intermax.to_numpy()-intermin.to_numpy()

        # df = test_data.get_miniOD()
        colstart = test_data.get_dep_cols(2015)
        colend = test_data.get_arr_cols(2015)

        def f(h):
            for i in range(len(self.hours)):
                if self.hours[i] > h:
                    return i - 1
            return len(self.hours) - 1

        df['period'] = df['Heure'].apply(f)
        # df = df.groupby(['Annee', 'Mois', 'Jour', 'wday', 'period']).sum()
        # df['period'] = list(map(lambda x: x[4], df.index.values))
        # df['wday'] = list(map(lambda x: x[3], df.index.values))
        grp = df.groupby(['period', 'wday'])
        l, m, M = [], [], []
        stations = list(map(lambda x: int(x[9:]), colend))
        cap = test_data.get_stations_capacities(None, stations)
        cap.index = list(map(lambda x: 'End date ' + str(x), cap.index.values))
        for ((p, wd), d) in grp:
            # d = d.set_index([list(map(lambda x:x[3]*24+x[4],d.index))])
            if single:
                d = d.iloc[0:self.length[p], :]
            inter = intermin.loc[24 * wd + self.hours[p], :]
            res_min = mse(maxi(d[colstart].to_numpy() - inter[colstart].to_numpy(), 0), 0)
            # print(res_min)
            inter = intermax.loc[24 * wd + self.hours[p], :]
            available_docks = maxi(cap - inter, 0)
            res_max = mse(maxi(d[colend].to_numpy() - available_docks[colend].to_numpy(), 0), 0)
            # print(res_max)
            m.append(res_min)
            M.append(res_max)
            if not arr_dep:
                l.append((res_max + res_min) / 2)
            elif arr_dep == 'arr':
                l.append(res_max)
            else:
                l.append(res_min)
        # print(np.array(m).mean())
        # print(np.array(M).mean())
        err = np.array(l).mean()
        return err

    def eval_target(self, test_data, distrib, intervals, df, single=False):
        """
        test the target value
        :param test_data: environment
        :param distrib: not used
        :param intervals: intervals data (3 df)
        :param df: the testing data
        :param single: to test on the first occurence of each period
        :return: test value
        """
        _, _, intertar = intervals
        intertar_start = intertar.copy()
        intertar_end = intertar
        intertar_start.columns = list(map(lambda x: x.replace('Tar', 'Start date'), intertar.columns.values))
        intertar_end.columns = list(map(lambda x: x.replace('Tar', 'End date'), intertar.columns.values))

        colstart = test_data.get_dep_cols(2015)  # [i for i in df.columns.values if 'Start date' in i]
        colend = test_data.get_arr_cols(2015)  # [i for i in df.columns.values if 'End date' in i]

        def f(h):
            for i in range(len(self.hours)):
                if self.hours[i] > h:
                    return i - 1
            return len(self.hours) - 1

        df['period'] = df['Heure'].apply(f)
        # df = df.groupby(['Annee','Mois','Jour','wday','period']).sum()
        # df['period']=list(map(lambda x:x[4],df.index.values))
        # df['wday']=list(map(lambda x:x[3],df.index.values))
        grp = df.groupby(['period', 'wday'])
        l = []
        m = []
        M = []
        stations = list(map(lambda x: int(x[9:]), colend))
        cap = test_data.get_stations_capacities(None, stations)
        cap.index = list(map(lambda x: 'End date ' + str(x), cap.index.values))
        for ((p, wd), d) in grp:
            if single:
                d = d.iloc[0:self.length[p], :]
            # d = d.set_index([list(map(lambda x:x[3]*24+x[4],d.index))])
            inter = intertar_start.loc[24 * wd + self.hours[p], :]
            res_min = mse(maxi(d[colstart].to_numpy() - inter[colstart].to_numpy(), 0), 0)
            # print('min',res_min)
            inter = intertar_end.loc[24 * wd + self.hours[p], :]
            available_docks = maxi(cap - inter, 0)
            res_max = mse(maxi(d[colend].to_numpy() - available_docks[colend].to_numpy(), 0), 0)
            # print('max',res_max)
            m.append(res_min)
            M.append(res_max)
            l.append((res_max + res_min) / 2)
        err = np.array(l).mean()
        # print(np.array(m).mean())
        # print(np.array(M).mean())
        return err

    @staticmethod
    def mean_interval_size(test_data, distrib, intervals):
        """
        computes average interval size
        :param test_data: environment
        :param distrib: not used
        :param intervals: intervals values
        :return:
        """
        intermin, intermax, intertar = intervals
        intermin.columns = list(map(lambda x: x.replace('Min', 'Start date'), intermin.columns.values))
        intermax.columns = list(map(lambda x: x.replace('Max', 'Start date'), intermax.columns.values))
        colstart = test_data.get_dep_cols(2015)
        return (intermax[colstart] - intermin[colstart]).to_numpy().mean()

    @staticmethod
    def sum_int(test_data, intervals):
        """
        computes the sum of min, max, and target values
        :param test_data: env
        :param intervals:
        :return: min, max, tar sums
        """
        intermin, intermax, intertar = intervals
        intermin.columns = list(map(lambda x: x.replace('Min', 'Start date'), intermin.columns.values))
        intermax.columns = list(map(lambda x: x.replace('Max', 'Start date'), intermax.columns.values))
        intertar.columns = list(map(lambda x: x.replace('Tar', 'Start date'), intertar.columns.values))
        colstart = test_data.get_dep_cols(2015)
        return int(intermin[colstart].to_numpy().sum()) / 35, int(intermax[colstart].to_numpy().sum()) / 35, int(
            intertar[colstart].to_numpy().sum()) / 35


if __name__ == '__main__':
    time_period = int(sys.argv[1])
    env = Environment('Bixi', 'train')
    data = Data(env)
    mod = ModelStations(env, 'svd', 'gbt', dim=10,**{'var':True})
    # mod = CombinedModelStation(env)
    mod.train(data)
    mod.save()
    mod.load()
    DI = DecisionIntervals(env, mod, 0.45, 0.6) #alpha and beta


    env = Environment('Bixi', 'test')
    data = Data(env)

    ########################################################Run once and compute the intervals according to the forecast of the next hours

    WH = mod.get_factors_forecast(data,time_period)
    #print(WH)
    intervals = DI.compute_decision_intervals(WH, data, predict=True)
    if os.path.exists(r'C:\Users\cmartins\Documents\intervals.csv'):
        int_df = pd.read_csv(r'C:\Users\cmartins\Documents\intervals.csv')
    else:
        int_df = pd.DataFrame(columns=['datetime', 'time_period', 'min', 'max', 'target'])
    int_df = int_df.append({'datetime': datetime.now(), 'time_period': time_period, 'min': ','.join([str(x) for x in intervals[0]]),
        'max': ','.join([str(x) for x in intervals[2]]), 'target': ','.join([str(x) for x in intervals[1]])}, ignore_index=True)
    int_df.to_csv(r'C:\Users\cmartins\Documents\intervals.csv', index=False)

    print("Normal Interval")
    print(intervals)

    ########################################################  Compute each hour separately and the intervals are a mean of each interval
    intervals_per_hour = [tuple(x) for x in np.zeros((3, 552), int)]
    if os.path.exists(r'C:\Users\cmartins\Documents\x.csv'):
        int_df = pd.read_csv(r'C:\Users\cmartins\Documents\x.csv')
    else:
        int_df = pd.DataFrame(columns=['datetime', 'hour', 'min', 'max', 'target'])

    for i, row in WH.iterrows():
        x = DI.compute_decision_intervals(WH.iloc[i], data, predict=True)
        intervals_per_hour = tuple(map(operator.add,intervals_per_hour, x))
        int_df = int_df.append({'datetime': datetime.now(), 'hour': row['Heure'], 'min': ','.join([str(x2) for x2 in x[0]]),
            'max': ','.join([str(x2) for x2 in x[2]]), 'target': ','.join([str(x2) for x2 in x[1]])}, ignore_index=True)
    value = [np.around(x / DI.length[time_period]) for x in intervals_per_hour]
    int_df.to_csv(r'C:\Users\cmartins\Documents\x.csv', index=False)

    if os.path.exists(r'C:\Users\cmartins\Documents\intervals_mean.csv'):
        int_df = pd.read_csv(r'C:\Users\cmartins\Documents\intervals_mean.csv')
    else:
        int_df = pd.DataFrame(columns=['datetime', 'time_period', 'min', 'max', 'target'])
    print("Mean Interval")
    print(value)
    int_df = int_df.append({'datetime': datetime.now(), 'time_period': time_period, 'min': ','.join([str(x) for x in value[0]]),
        'max': ','.join([str(x) for x in value[2]]), 'target': ','.join([str(x) for x in value[1]])}, ignore_index=True)
    int_df.to_csv(r'C:\Users\cmartins\Documents\intervals_mean.csv', index=False)
    

