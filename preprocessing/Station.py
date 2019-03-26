import numpy as np
import pandas as pd


class Stations(object):
    def __init__(self, env, since=2015):
        self.s = pd.read_csv(open(env.station_info), delimiter=',')
        self.s.sort_values(['since', 'code'], axis=0, inplace=True)
        self.s = self.s.loc[self.s['used'] == 1, :]
        self.s['pk'].astype(int, inplace=True)
        self.s['code'].astype(int, inplace=True)
        self.s.index = self.s['code']
        # self.pk_to_id = None
        self.since = since

    def get_station_number(self, since=None):
        if since:
            return (self.s['since']==since).sum()
        else :
            return self.s.shape[0]

    def get_since(self):
        res = self.s['since'].to_numpy().flatten()
        res = res.astype(int)
        return res

    def get_ids(self):
        if self.since is None:
            res = self.s['code'].to_numpy().flatten()
        else:
            # print(self.)
            res = self.s['code'][self.s['since'] == self.since].to_numpy().flatten()
        res = res.astype(int)
        return res

    def get_pks(self):
        if self.since is None:
            return self.s['pk'].to_numpy().flatten()
        else:
            return self.s['pk'][self.s['since'] == self.since].to_numpy().flatten()

    def get_id_from_pk(self, pk):
        # if self.pk_to_id is None:
        #     self.pk_to_id = {}
        #     for i in range(self.s.shape[0]):
        #         self.pk_to_id[self.get_pks()[i]] = self.get_ids()[i]
        self.s.index = self.s['pk']
        try:
            res = self.s.loc[pk, 'code']
            self.s.index = self.s['code']
            return res
        except KeyError:
            self.s.index = self.s['code']
            return None

    def get_id_from_name(self, name):
        # if self.pk_to_id is None:
        #     self.pk_to_id = {}
        #     for i in range(self.s.shape[0]):
        #         self.pk_to_id[self.get_pks()[i]] = self.get_ids()[i]
        self.s.index = self.s['name']
        try:
            res = self.s.loc[name, 'code']
            self.s.index = self.s['code']
            return res
        except KeyError:
            self.s.index = self.s['code']
            return None

    def get_loc(self):
        if self.since is None:
            return self.s[['lng', 'lat']].apply(pd.to_numeric)
        else:
            res = (self.s[['lng', 'lat']][self.s['since'] == self.since]).apply(pd.to_numeric)
            return res

    def get_capacities(self, tab=None):
        self.s.index = self.s['code'].astype(int)
        if self.since is None:
            b = self.s['capacity'] > 0
        else:
            b = self.s['since'] == self.since
        if tab is None:
            res = self.s['capacity'][b]
            return res.astype(int)
        else:
            tab = np.intersect1d(tab, self.s.index.values, assume_unique=True)
            res = self.s.loc[tab, 'capacity'][b[tab]]
            return res

    def get_mtm(self):
        if self.since is None:
            return self.s[['xmtm', 'ymtm']]
        else:
            res = self.s[['xmtm', 'ymtm']][self.s['since'] == self.since]
            return res


if __name__ == '__main__':
    from preprocessing.Environment import Environment

    env = Environment('Bixi', 'train')
    s = Stations(env, None)
    s.get_id_from_pk(456)
    print(str(s.get_ids()))
