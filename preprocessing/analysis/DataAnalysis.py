import math

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.modelUtils import *
from preprocessing.Data import Data


class DataAnlysis(Data):
    """
    super class of Data, to analyse the data
    gather all display function on the data 
    """

    def __init__(self, env):
        super(DataAnlysis, self).__init__(env)

    def plot_week(self, station=None):
        """
        displays the mean week of the designated station 
        :param station: station to be displayed, if None then the mean of the total number of trips is displayed
        :return: none
        """
        if station is None:
            data = self.get_ODsum()
            grp = data.groupby(['wday', 'Heure'])['End date'].mean()
            grp.index = range(grp.shape[0])
            plt.plot(grp)
            plt.show()
        else:
            data = self.get_miniOD([])
            grp = data.groupby(['wday', 'Heure'])['End date ' + str(station)].mean()
            grp.index = range(grp.shape[0])
            plt.plot(grp)
            grp = data.groupby(['wday', 'Heure'])['Start date ' + str(station)].mean()
            grp.index = range(grp.shape[0])
            plt.plot(grp)
            plt.show()

    def tezst(self):
        data = self.get_miniOD([])
        arr = self.get_arr_cols(None)
        dep = self.get_dep_cols(None)
        narr = data[arr].to_numpy().sum()
        ndep = data[arr].to_numpy().sum()
        # print(narr)
        # print(ndep)
        # print(ndep - narr)
        # print(np.abs((data[arr].to_numpy() - data[dep].to_numpy())).mean())
        # d = data['End date']-data['Start date']
        # print(data['End date'].mean())
        # print(d.mean())

    def scatter_plots(self, learning_var, station=None):
        """
        displays the dependency between trip numbers and properties  
        :param learning_var: field of the analysis
        :param station: the station of the analysis, if none analysis performed on the total number of trips
        :return: none
        """
        from config import translate_var
        if station is None:
            data = self.get_ODsum()
        else:
            data = self.get_miniOD([])
        ind = data[data['annee'] == 0].index
        data.drop(ind, inplace=True)
        print(data.columns.values)
        i = 0
        if station is None:
            ch_an = 'End date'
        else:
            ch_an = 'End date ' + str(station)
        for ch in learning_var:
            if not (ch[0] == 'h') and not (ch in ['LV', 'MMJ', 'SD', 'poudrerie', 'verglas','total']):
                # data.boxplot(ch_an, by=ch)
                # if ch != 'Heure':
                plt.figure(i // 9,figsize=(15,7))
                plt.subplots_adjust(hspace=0.36)
                fig = plt.subplot(3, 3, (i % 9) + 1)
                i += 1
                # fig = plt.figure().add_subplot(111)
                fig.set_xlabel(translate_var[ch])
                fig.set_ylabel('trip number')
                l = []
                xaxis = np.unique(data[ch])
                print(ch, xaxis.shape)
                if xaxis.shape[0] < 20 or ch == 'Heure':
                    for u in xaxis:
                        l.append(data[ch_an][data[ch] == u])
                else:
                    m = np.min(data[ch])
                    M = np.max(data[ch])
                    step = (M - m) / 20
                    xaxis = np.arange(m, M, step)
                    for u in xaxis:
                        l.append(data[ch_an][(data[ch] >= u) * (data[ch] < u + step)])
                xaxis = xaxis.astype(int)
                # fig = plt.boxplot(ch_an, by=ch)
                # g = data.groupby(ch).mean()[ch_an]
                # v = data.groupby(ch).std()[ch_an]
                plt.boxplot(l, labels=xaxis)
        for i in range(3):
            plt.figure(i)
            plt.savefig(config.img_path+'data_'+str(i)+'.pdf', bbox_inches='tight')
                # plt.plot(g, '-r')
                # plt.plot(g + v, ':r')
                # plt.plot(g - v, ':r')
        plt.show()

    def t_tests(self, station=None):
        """
        test the significance of the variables on the objective 
        :param station: the station of the analysis, if none analysis performed on the total number of trips 
        :return: None
        """
        if station is None:
            data = self.get_ODsum()
        else:
            data = self.get_miniOD([])
            data.rename({'End date ' + str(station): 'End date', 'Start date ' + str(station): 'Start date'})
        ind = data[data['Annee'] == 0].index
        data.drop(ind, inplace=True)
        variable01 = list(data.columns.values)
        l = self.get_stations_col(2015)
        l.append('End date')
        l.append('Start date')
        for i in l:
            try:
                variable01.remove(i)
            except:
                pass
        multi_var = ['Annee', 'Mois', 'Jour', 'Heure', 'wday']

        continous_var = ['pression', 'temp', 'Hum', 'vent', 'visi']

        for ch in multi_var:
            try:
                variable01.remove(ch)
            except:
                pass
        for ch in continous_var:
            try:
                variable01.remove(ch)
            except:
                pass
        from scipy.stats import ttest_ind
        for ch in continous_var:
            print(ch)
            m = data[ch].min()
            M = data[ch].max()
            for i in range(int(m), int(M), max(1, int((M - m) / 10))):
                d = data[ch] <= i
                print(ttest_ind(data['End date'][d], data['End date'][1 - d]),
                      ttest_ind(data['Start date'][d], data['Start date'][1 - d]))
        for ch in multi_var:
            print(ch)
            u = np.unique(data[ch].to_numpy())
            for i in u:
                d = data[ch] == i
                print(ttest_ind(data['End date'][d], data['End date'][1 - d]),
                      ttest_ind(data['Start date'][d], data['Start date'][1 - d]))
        for ch in variable01:
            d = data[ch] == 0
            # d = np.random.binomial(1, 0.5, data[ch].shape)
            print(ch)
            print(ttest_ind(data['End date'][d], data['End date'][1 - d]),
                  ttest_ind(data['Start date'][d], data['Start date'][1 - d]))

    def traffic_hour_mean_dependency(self, learning_var, station=None):
        """
        displays the influence of each parameter on the trip numbers per hour
        :param learning_var: fields where to compute the analysis
        :param station: the station of the analysis, if none analysis performed on the total number of trips
        :return: None
        """
        if station is None:
            data = self.get_ODsum()
            ch_an = ['End date', 'Start date']
        else:
            data = self.get_miniOD([])
            ch_an = ['End date ' + str(station), 'Start date ' + str(station)]
        values = {
            'Heure': 24,
            "temp": 10,
            "Vit. du vent (km/h)": 5,
            "visi": 10,
            "Mois": 8,
            "Jour": 10,
            "wday": 6,
            "ferie": 1,
            "averses": 1,
            "precip": 1,
            "fort": 1,
            "nuageux": 1,
            "brr": 1,
            "pression": 10,
            "vac": 1,
            "Annee": 1,
            "Hum": 10,
            "neige": 1,
            "pluie": 1,
            "verglas": 1,
            "bruine": 1,
            "poudrerie": 1,
            "brouillard": 1,
            "orage": 1,
            'LV': 1,
            'MMJ': 1,
            'SD': 1,
            'h0': 1,
            'h1': 1,
            'h2': 1,
            'h3': 1,
            'h4': 1,
            'h5': 1,
            'h6': 1,
            'h7': 1,
            'h8': 1,
            'h9': 1,
            'h10': 1,
            'h11': 1,
            'h12': 1,
            'h13': 1,
            'h14': 1,
            'h15': 1,
            'h16': 1,
            'h17': 1,
            'h18': 1,
            'h19': 1,
            'h20': 1,
            'h21': 1,
            'h22': 1,
            'h23': 1,
        }
        data = data[data['Mois'].isnull() == 0]
        x = y = int(math.sqrt(len(ch_an))) + 1
        j = 0
        for ch in learning_var:
            if (ch != 'Heure') and (ch != 'verglas') and (ch != 'poudrerie') and (ch != 'vac') and (ch[0] != 'h'):
                print(ch)
                plt.figure(j)
                j += 1
                k = 0
                if ch == 'pression':
                    data.loc[data[ch] == 0, ch] = 100
                M = data[ch].max()
                m = data[ch].min()

                data.loc[:, ch] = data.loc[:, ch].apply(lambda x: int((x - m) / (M - m) * values[ch]))
                if (M != m):
                    for an in ch_an:
                        fig = plt.subplot(x, y, (k % (x * y)) + 1)
                        k += 1
                        plt.title(ch)
                        fig.set_ylabel('trip number')
                        fig.set_xlabel('Heure')
                        g = data.groupby([ch, 'Heure']).mean()
                        g['Heure'] = list(map(lambda x: x[1], g.index.values))
                        g[ch] = list(map(lambda x: x[0], g.index.values))
                        # print(np.array(g[ch]))
                        h = []
                        colormap = plt.get_cmap('jet')
                        for i, grp in g.groupby(ch):
                            # print(grp)
                            h += plt.plot(grp['Heure'], np.array(grp[an]), label=i, c=colormap(i / values[ch]))
                        plt.legend()
        plt.show()

    def correlation(self, m=None):
        """
        compute and plots the correlation matrix
        :return: None
        """
        if m is None:
            m = self.get_ODsum()
        # try :
        m.drop(['verglas', 'poudrerie'], inplace=True, axis=1)
        # except ValueError:
        #     pass
        print(m.columns.values)
        plot_corr(m)
        from statsmodels.regression.linear_model import OLS
        y = m['End date'].to_numpy()
        x = m.drop(['End date', 'Start date'], axis=1).to_numpy()
        x = x.astype(int)
        lin = OLS(y, x)
        res = lin.fit()
        print(res.summary())
        plt.savefig(config.img_path+'corr.pdf', bbox_inche='tight')
        plt.show()

    def r_squared(self, station=None):
        """
        compute r2 score on linear regression 
        :param station: the station of the analysis, if none analysis performed on the total number of trips
        :return: None
        """
        if station is None:
            m = self.get_ODsum()
            y = m[['Start date', 'End date']]
            x = m.drop(['Start date', 'End date'], axis=1)
        else:
            m = self.get_miniOD([])
            y = m[self.get_stations_col(2015)]
            x = m.drop(self.get_stations_col(2015), axis=1)
        from sklearn.linear_model import LinearRegression
        lin = LinearRegression(n_jobs=-1)
        lin.fit(x, y)
        pred = lin.predict(x)
        from utils.modelUtils import r_squared
        print(r_squared(y, pred))

        # def r_squared_AD(self):
        #     m = self.get_ODsum()
        #     l = ['Start date', 'End date']
        #     y = m[l]
        #     x = m.drop(l, axis=1)
        #     from sklearn.linear_model import LinearRegression
        #     lin = LinearRegression(n_jobs=-1)
        #     lin.fit(x, y)
        #     pred = lin.predict(x)
        #     from utils.modelUtils import r_squared
        #     print(r_squared(y, pred))
        #     return

    def significancetestanova(self, station=None):
        if station is None:
            data = self.get_ODsum()
            data.rename(columns={'End date': 'ch'}, inplace=True)
            # ch = 'End date'
        else:
            data = self.get_miniOD([])
            data.rename(columns={'End date ' + str(station): 'ch'}, inplace=True)
        l = [x for x in data.columns.values if not (x.__contains__('date'))]
        # l.remove('verglas')
        # l.remove('poudrerie')
        for s in ['ch', 'poudrerie',
                  'verglas']:  # , 'precip', 'brr', 'LV', 'MMJ', 'SD', 'verglas', 'poudrerie', 'UTC timestamp']:
            l.remove(s)
        i = int(data.shape[0] * 0.8)
        train = data[:i]
        test = data[i:]
        sum = train[l].to_numpy().sum(axis=0)
        co = []
        pca = PCA()
        # pca.fit(train[l].to_numpy() / sum)
        # comp = pca.components_
        # print(comp.shape)
        # for n in range(1, 200):
        #     # for h in range(24):
        #     #     l.remove('h' + str(h))
        #     m=comp.copy()
        #     m *= n
        #     m = np.round(m)
        #     m /= n
        #     pca.components_ = m
        #     # p = pca.fit_transform(data[l].to_numpy() / sum)
        #     p = pca.transform(test[l].to_numpy() / sum)
        #     p = pd.DataFrame(p, columns=l)
        #     corr = p.corr().to_numpy()
        #     # corr = corr[1-np.isnan(corr)]
        #     c = np.nansum(np.abs(corr))
        #
        #     nan = np.isnan(corr[range(corr.shape[0]), range(corr.shape[0])]).sum()
        #     print(n,c, nan)
        #     co.append(c+50*nan)
        # plt.plot(co)
        # plt.show()
        n=18
        pca.fit(data[l].to_numpy() / sum)
        m = pca.components_
        m *= n
        m = np.round(m)
        m /= n
        pca.components_ = m
        # p = pca.fit_transform(data[l].to_numpy() / sum)
        p = pca.transform(data[l].to_numpy() / sum)
        p = pd.DataFrame(p)
        # p = data[l] / sum
        # print(pca.components_)
        p['intercept'] = 1
        # print(formula)
        y = data['ch'].to_numpy().flatten()
        # x = data[l].to_numpy().astype(float)
        # ls = ols(formula, data).fit()

        from statsmodels.regression.linear_model import OLS
        # y = m['End date'].to_numpy()
        x = p
        x = x.astype(float)
        lin = OLS(y, x)
        res = lin.fit()
        print(res.summary())


def plot_corr(df, size=10):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    :param df: pandas DataFrame
    :param size: vertical and horizontal size of the plot
    :return: None
    """
    df.rename(columns=config.translate_var, inplace=True)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    from matplotlib import cm
    cmap = cm.bwr
    ax.matshow(corr, cmap=cmap)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)


if __name__ == '__main__':
    from preprocessing.Environment import Environment
    from config import available_learning_var

    # available_learning_var

    ud = Environment('Bixi', 'train')
    # DataAnlysis(ud).scatter_plots(available_learning_var)
    DataAnlysis(ud).correlation()
    # DataAnlysis(ud).significancetestanova()
    # DataAnlysis(ud).plot_week()
    # DataAnlysis(ud).traffic_hour_mean_dependency(learning_var, station=None)
    # DataAnlysis(ud).r_squared()
