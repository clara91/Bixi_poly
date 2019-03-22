import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from config import root_path


def alpha():
    df = pd.read_csv(root_path + 'di_scores.csv')
    df['mean_alerts'] = df[['mean_alerts_dep', 'mean_alerts_arr']].mean(axis=1)
    df_alpha = df[df['beta'] == 0.65]
    df_alpha.sort_values('alpha', inplace=True)
    plt.figure(figsize=(9, 3))
    bixi = df[df['DI'] == 'bixi']
    cmap = cm.Set1
    for i, ch in enumerate(['lost_arr', 'lost_dep', 'eval_target']):
        plt.subplot(121)
        plt.plot(df_alpha['alpha'], df_alpha[ch], color=cmap(i / 9), linestyle='-', marker='o', label=ch)
        plt.xlabel('\u03B1')
        plt.ylabel('Lost trips Worst case')
        plt.plot([0, 1], [bixi[ch], bixi[ch]], color=cmap(i / 9), linestyle=':', marker='', label=ch + '_Bixi')
    plt.legend()
    for i, ch in enumerate(['mean_alerts_dep', 'mean_alerts_arr', 'mean_alerts']):
        plt.subplot(122)
        plt.plot(df_alpha['alpha'], df_alpha[ch], linestyle='-', marker='o', label=ch)
        plt.xlabel('\u03B1')
        plt.ylabel('Number of alerts per hour')
        plt.plot([0, 1], [bixi[ch], bixi[ch]], color=cmap(i / 9), linestyle=':', marker='', label=ch + '_Bixi')
    plt.legend()
    # for i, ch in enumerate(['mean_size']):
    #     plt.subplot(133)
    #     plt.plot(df_alpha['alpha'], df_alpha[ch], linestyle='-', marker='o', label=ch)
    #     plt.xlabel('\u03B1')
    #     plt.ylabel('Interval average sizes')
    #     plt.plot([0, 1], [bixi[ch], bixi[ch]], color=cmap(i / 9), linestyle=':', marker='', label=ch + '_Bixi')
    plt.legend()
    plt.savefig('alpha2.pdf', bbox_inches='tight')
    plt.show()


alpha()


def beta():
    df = pd.read_csv(root_path + 'di_scores.csv')
    df['mean_alerts'] = df[['mean_alerts_dep', 'mean_alerts_arr']].mean(axis=1)
    df_beta = df[df['alpha'] == 0.5]
    df_beta.sort_values('beta', inplace=True)
    plt.figure(figsize=(9, 3))
    bixi = df[df['DI'] == 'bixi']
    cmap = cm.Set1
    for i, ch in enumerate(['lost_arr', 'lost_dep', 'eval_target']):
        plt.subplot(121)
        plt.plot(df_beta['beta'], df_beta[ch], color=cmap(i / 9), linestyle='-', marker='o', label=ch)
        plt.xlabel('\u03B2')
        plt.ylabel('Lost trips Worst case')
        plt.plot([0, 1], [bixi[ch], bixi[ch]], color=cmap(i / 9), linestyle=':', marker='', label=ch + '_Bixi')
    plt.legend()
    for i, ch in enumerate(['mean_alerts_dep', 'mean_alerts_arr', 'mean_alerts']):
        plt.subplot(122)
        plt.plot(df_beta['beta'], df_beta[ch], linestyle='-', marker='o', label=ch)
        plt.xlabel('\u03B2')
        plt.ylabel('Number of alerts per hour')
        plt.plot([0, 1], [bixi[ch], bixi[ch]], color=cmap(i / 9), linestyle=':', marker='', label=ch + '_Bixi')
    plt.legend()
    # for i, ch in enumerate(['mean_size']):
    #     plt.subplot(133)
    #     plt.plot(df_beta['beta'], df_beta[ch], linestyle='-', marker='o', label=ch)
    #     plt.xlabel('\u03B2')
    #     plt.ylabel('Interval average sizes')
    #     plt.plot([0, 1], [bixi[ch], bixi[ch]], color=cmap(i / 9), linestyle=':', marker='', label=ch + '_Bixi')
    # plt.legend()
    plt.savefig('beta2.pdf', bbox_inches='tight')
    plt.show()


beta()


def alerts_vs_lost_trips():
    df = pd.read_csv(root_path + 'di_scores.csv')
    df['mean_alerts'] = df[['mean_alerts_dep', 'mean_alerts_arr']].mean(axis=1)
    df_beta = df[df['alpha'] == 0.5]
    df_beta.sort_values('beta', inplace=True)
    bixi = df[df['DI'] == 'bixi']
    cmap = cm.Paired
    plt.figure(figsize=(10, 7))
    plt.plot(df_beta['mean_alerts_dep'], df_beta['lost_dep'], color=cmap(1 / 12), marker='o', label='Departures')
    plt.plot(bixi['mean_alerts_dep'], bixi['lost_dep'], color=cmap(1 / 12), marker='x', label='Bixi Dep',
             linestyle='None')
    plt.plot(df_beta['mean_alerts_arr'], df_beta['lost_arr'], color=cmap(3 / 12), marker='o', label='Arrivals')
    plt.plot(bixi['mean_alerts_arr'], bixi['lost_arr'], color=cmap(3 / 12), marker='x', label='Bixi Arr',
             linestyle='None')
    plt.legend()
    plt.xlabel('Number of alerts per hour')
    plt.ylabel('Number of trip lost per hour')
    plt.savefig('alerts_vs_lost_trips.pdf', bbox_inches='tight')
    plt.show()


# alpha()
# beta()
