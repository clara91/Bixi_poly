import pandas as pd
import numpy as np
import time
import config
import datetime
import matplotlib.pyplot as plt
from preprocessing.Environment import Environment


def analyse_gaps(env):
    files = env.get_meteo_files()
    m = {'ï»¿"Date/Heure"': 'Date/Heure',
         'AnnÃ©e': 'Annee',
         'QualitÃ© des DonnÃ©es': 'qualite donnee',
         'Temp (Â°C)': 'temp',
         'Point de rosÃ©e (Â°C)': 'point de rosee',
         'Point de rosÃ©e Indicateur': 'point de rosee ind',
         'VisibilitÃ© (km)': 'visi',
         'VisibilitÃ© Indicateur': 'visi ind',
         'Pression Ã\xa0 la station (kPa)': 'pression',
         'Pression Ã\xa0 la station Indicateur': 'pression ind',
         'Refroid. Ã©olien': 'eolien',
         'Refroid. Ã©olien Indicateur': 'eolien ind',
         'Hum. rel (%)': 'Hum',
         'Vit. du vent (km/h)': 'vent'
         }
    frames = []
    for y in files.keys():
        for file in files[y]:
            try:
                path = env.precipitation_path
                df = pd.read_csv(open(path + file), delimiter=',', quotechar='"')
                df.rename(index=str, columns=m, inplace=True)
                df['UTC timestamp'] = df['Date/Heure'].apply(
                    lambda x: time.mktime(datetime.datetime.strptime(str(x)[:13], "%Y-%m-%d %H").timetuple()))
                frames.append(df)
            except FileNotFoundError:
                print('file not found : ' + file)
                pass
    r = pd.concat(frames)
    r['Heure'] = r['Heure'].apply(lambda x: int(x[:2]))
    for c in ['temp', 'visi', 'pression']:
        r[c] = r[c].apply(lambda x: float(str(x).replace(',', '.')))
    # r.loc[r['pression'] == 0, 'pression'] = np.nan

    # r.loc[r['pression'].isnull(), 'pression'] = np.nanmean(r['pression'].as_matrix())
    ferie = pd.read_csv(env.off_days, delimiter=',', quotechar='"')
    # ferie['jour'] = ferie['\xef\xbb\xbfjour']
    ferie['ferie'] = np.ones(ferie.shape[0], dtype=int)
    # ferie.drop('\xef\xbb\xbfjour', inplace=True, axis=1)
    def daycode(year, month,day):
        return 380*(year-2010)+31*month+day
    ferie['daycode'] = daycode(ferie['annee'],ferie['mois'],ferie['jour'])
    ferie.drop(['jour', 'annee', 'mois'], inplace=True, axis=1)
    r['daycode'] = daycode(r['Annee'],r['Mois'],r['Jour'])
    r = pd.merge(r, ferie, how='left', on='daycode')
    r['period']=(r['daycode']>daycode(2015,4,15))*(r['daycode']<daycode(2015,11,15))+(r['daycode']>daycode(2016,4,15))*(r['daycode']<daycode(2016,11,15))
    r=r[r['period']]



    r.drop('daycode', inplace=True, axis=1)
    r.loc[r['ferie'].isnull(), 'ferie'] = 0
    r['averses'] = r['Temps'].apply(lambda x: int(str(x).lower().find('averses') != -1))
    r['neige'] = r['Temps'].apply(lambda x: int(str(x).lower().find('neige') != -1))
    r['pluie'] = r['Temps'].apply(lambda x: int(str(x).lower().find('pluie') != -1))
    r['fort'] = r['Temps'].apply(lambda x: int(str(x).lower().find('fort') != -1))
    r['modere'] = r['Temps'].apply(lambda x: int(str(x).lower().find('modã©rã©e') != -1))
    r['verglas'] = r['Temps'].apply(lambda x: int(str(x).lower().find('vergla') != -1))
    r['bruine'] = r['Temps'].apply(lambda x: int(str(x).lower().find('bruine') != -1))
    r['poudrerie'] = r['Temps'].apply(
        lambda x: int(str(x).lower().find('poudrerie') != -1) + int(str(x).lower().find('granules') != -1))
    r['brouillard'] = r['Temps'].apply(
        lambda x: int(str(x).lower().find('brouillard') != -1) + int(str(x).lower().find('brume') != -1))
    r['nuageux'] = r['Temps'].apply(lambda x: int(str(x).lower().find('nuageux') != -1))
    r['orage'] = r['Temps'].apply(lambda x: int(str(x).lower().find('orage') != -1))
    r['degage'] = r['Temps'].apply(lambda x: int(str(x).lower().find('dã©gagã©') != -1))

    r['precip'] = r['pluie'] | r['neige'] | r['averses'] | r['orage']
    r['brr'] = r['brouillard'] | r['bruine']
    r['ND'] = r['Temps'].apply(lambda x: int(str(x).lower().find('nd') != -1))
    l = config.learning_var
    l.remove('Heure')
    l.remove('Mois')
    l.remove('Annee')
    l.remove('wday')
    l.remove('Jour')
    ll = ['neige', 'pluie', 'fort', 'modere', 'verglas', 'bruine', 'poudrerie', 'brouillard', 'nuageux', 'orage',
          'degage','ND' ]
    a = r[ll].as_matrix().sum(axis=0)
    print(r.shape)
    print(r['temp'].isnull().sum())
    print(r['Hum'].isnull().sum())
    print((r['pression']==0).sum())
    print(r['pression'].isnull().sum())
    print(r['vent'].isnull().sum())
    print(r['visi'].isnull().sum())
    print(r['eolien'].isnull().sum())
    print((r['Temps']=='ND').sum())
    print(r[['temp','vent','Hum','visi','UTC timestamp']][r['temp'].isnull()])
    plt.bar(range(len(a)),a)
    plt.xticks(range(len(ll)), ll, rotation='vertical')
    plt.show()

    a = np.zeros(r['Temps'].shape,dtype=int)
    k=0
    for i in range(a.shape[0]):
        if r['Temps'].iloc[i]=='ND':
            a[i]=a[i-1]+1
        else:
            k+=int(r['Temps'].iloc[i]==r['Temps'].iloc[i-1-a[i-1]])
    print('k', k)
    plt.hist(a)
    for i in range(7):
        print((a==i).sum())
    print(np.bincount(a))
    print((a!=0).sum())
    plt.show()

    plt.hist(r['Heure'])
    # plt.boxplot(r['temp']),
    # plt.show()

def compute_pre_heure(env):
    """
    computes the matrix with all weather features per hour
    :param env: 
    :return: 
    """
    files = env.get_meteo_files()
    m = {'ï»¿"Date/Heure"': 'Date/Heure',
         'AnnÃ©e': 'Annee',
         'QualitÃ© des DonnÃ©es': 'qualite donnee',
         'Temp (Â°C)': 'temp',
         'Point de rosÃ©e (Â°C)': 'point de rosee',
         'Point de rosÃ©e Indicateur': 'point de rosee ind',
         'VisibilitÃ© (km)': 'visi',
         'VisibilitÃ© Indicateur': 'visi ind',
         'Pression Ã\xa0 la station (kPa)': 'pression',
         'Pression Ã\xa0 la station Indicateur': 'pression ind',
         'Refroid. Ã©olien': 'eolien',
         'Refroid. Ã©olien Indicateur': 'eolien ind',
         'Hum. rel (%)': 'Hum',
         'Vit. du vent (km/h)': 'vent'
         }

    def load(name):
        path = env.precipitation_path
        df = pd.read_csv(open(path + name), delimiter=',', quotechar='"')
        df.rename(index=str, columns=m, inplace=True)
        df['UTC timestamp'] = df['Date/Heure'].apply(
            lambda x: time.mktime(datetime.datetime.strptime(str(x)[:13], "%Y-%m-%d %H").timetuple()))
        k = 0
        li = ['Temps', 'vent', 'temp', 'visi',
              'pression', 'Hum']
        while k < 5:
            k = k + 1
            h = 'Temps'
            b = df[h] == 'ND'
            df.loc[b, h] = df[h].shift(1)[b]
            for h in li:
                b = df[h].isnull()
                df.loc[b, h] = df[h].shift(1)[b]
        k = 0
        while k < 5:
            k = k + 1
            h = 'Temps'
            b = df[h] == 'ND'
            df.loc[b, h] = df[h].shift(-1)[b]
            for h in li:
                b = df[h].isnull()
                df.loc[b, h] = df[h].shift(-1)[b]
        return df

    frames = []
    for y in files.keys():
        for file in files[y]:
            try:
                frames.append(load(file))
            except FileNotFoundError:
                print('file not found : ' + file)
                pass
    r = pd.concat(frames)
    ##########################################
    ##       cast string to numbers         ##
    ##########################################
    r['Heure'] = r['Heure'].apply(lambda x: int(x[:2]))
    for c in ['temp', 'visi', 'pression']:
        r[c] = r[c].apply(lambda x: float(str(x).replace(',', '.')))
    r.loc[r['pression'] == 0, 'pression'] = np.nan

    r.loc[r['pression'].isnull(), 'pression'] = np.nanmean(r['pression'].as_matrix())

    ##########################################
    ##    chargement des jours feries       ##
    ##########################################
    ferie = pd.read_csv(env.off_days, delimiter=',', quotechar='"')
    # ferie['jour'] = ferie['\xef\xbb\xbfjour']
    ferie['ferie'] = np.ones(ferie.shape[0], dtype=int)
    # ferie.drop('\xef\xbb\xbfjour', inplace=True, axis=1)
    ferie['daycode'] = 380 * (ferie['Annee'] - 2000) + ferie['mois'] * 31 + ferie['jour']
    ferie.drop(['jour', 'Annee', 'mois'], inplace=True, axis=1)
    r['daycode'] = 380 * (r['Annee'] - 2000) + r['Mois'] * 31 + r['Jour']
    r = pd.merge(r, ferie, how='left', on='daycode')
    r.drop('daycode', inplace=True, axis=1)
    r.loc[r['ferie'].isnull(), 'ferie'] = 0
    ##########################################
    ##      chargement des vacances         ##
    ##########################################
    vac = pd.read_csv(env.holidays, delimiter=',', quotechar='"')
    # vac['jour'] = vac['\xef\xbb\xbfjour']
    vac['vac'] = np.ones(vac.shape[0], dtype=int)
    # vac.drop('\xef\xbb\xbfjour', inplace=True, axis=1)
    vac['daycode'] = 380 * (vac['Annee'] - 2000) + vac['mois'] * 31 + vac['jour']
    vac.drop(['jour', 'Annee', 'mois'], inplace=True, axis=1)
    r['daycode'] = 380 * (r['Annee'] - 2000) + r['Mois'] * 31 + r['Jour']
    r = pd.merge(r, vac, how='left', on='daycode')
    r.drop('daycode', inplace=True, axis=1)
    r.loc[r['vac'].isnull(), 'vac'] = 0
    ####################################################
    ##  transformation du temps n colonnes binaires   ##
    ####################################################
    r['averses'] = r['Temps'].apply(lambda x: int(str(x).lower().find('averses') != -1))
    r['neige'] = r['Temps'].apply(lambda x: int(str(x).lower().find('neige') != -1))
    r['pluie'] = r['Temps'].apply(lambda x: int(str(x).lower().find('pluie') != -1))
    r['fort'] = r['Temps'].apply(lambda x: int(str(x).lower().find('fort') != -1))
    r['modere'] = r['Temps'].apply(lambda x: int(str(x).lower().find('modã©rã©e') != -1))
    r['verglas'] = r['Temps'].apply(lambda x: int(str(x).lower().find('vergla') != -1))
    r['bruine'] = r['Temps'].apply(lambda x: int(str(x).lower().find('bruine') != -1))
    r['poudrerie'] = r['Temps'].apply(
        lambda x: int(str(x).lower().find('poudrerie') != -1) + int(str(x).lower().find('granules') != -1))
    r['brouillard'] = r['Temps'].apply(
        lambda x: int(str(x).lower().find('brouillard') != -1) + int(str(x).lower().find('brume') != -1))
    r['nuageux'] = r['Temps'].apply(lambda x: int(str(x).lower().find('nuageux') != -1))
    r['orage'] = r['Temps'].apply(lambda x: int(str(x).lower().find('orage') != -1))
    r['degage'] = r['Temps'].apply(lambda x: int(str(x).lower().find('dã©gagã©') != -1))

    r['precip'] = r['pluie'] | r['neige'] | r['averses'] | r['orage']
    r['brr'] = r['brouillard'] | r['bruine']
    ####################################################
    ##            temperature ressentie               ##
    ####################################################
    # r['tempress'] = r['temp'] + r['eolien']
    r.to_pickle(env.pre_per_hour_path)
    return r

if __name__ == '__main__':
    analyse_gaps(Environment('Bixi','train'))