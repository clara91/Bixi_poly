import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from model_station.CombinedModelStation import CombinedModelStation
from decision_intervals.decision_intervals import *
from preprocessing.preprocess import get_features


def eval_worst_case(test_data, distrib, path, arr_dep=None):
    """
    computes the worst case test value
    :param test_data: env and data
    :param distrib:
    :param path: path of the intervals
    :param arr_dep: chose test ('dep' 'arr' or '')
    :return: test values
    """
    DI = DecisionIntervals(test_data.env, None, 0.5, None)
    return DI.eval_worst_case(test_data, distrib, DI.load_intervals(test_data, path, distrib), test_data.get_miniOD([]),
                              arr_dep=arr_dep)


def eval_target(test_data, distrib, path):
    """
    computes the target test value
    :param test_data: env and data
    :param distrib:
    :param path: path of the intervals
    :return: test values
    """
    DI = DecisionIntervals(test_data.env, None, 0.5, None)
    return DI.eval_target(test_data, distrib, DI.load_intervals(test_data, path, distrib), test_data.get_miniOD([]))


def mean_interval_size(test_data, distrib, path):
    """
    computes the average interval size
    :param test_data: env and data
    :param distrib:
    :param path: path if intervals
    :return: average interval size
    """
    DI = DecisionIntervals(test_data.env, None, 0.5, None)
    return DI.mean_interval_size(test_data, distrib,
                                 DI.load_intervals(test_data, path, distrib))  # , test_data.get_miniOD())


def mean_interval_size_percent(test_data, distrib, path):
    """
    computes the average interval size (percentages)
    :param test_data: env and data
    :param distrib:
    :param path: path if intervals
    :return: average interval size
    """
    DI = DecisionIntervals(test_data.env, None, 0.5, None)
    return DI.compute_mean_intervals(test_data, path, distrib)  # , test_data.get_miniOD())


def sum_int(test_data, distrib, path):
    """
    computes the average interval size
    :param test_data: env and data
    :param distrib:
    :param path: path if intervals
    :return: average interval size
    """
    DI = DecisionIntervals(test_data.env, None, 0.5, None)
    return DI.sum_int(test_data, DI.load_intervals(test_data, path, distrib))  # , test_data.get_miniOD())


def mean_alerts(test_data, distrib, path):
    """
    computes the average number of alerts per day
    :param test_data: env and data
    :param distrib:
    :param path: path if intervals
    :return: average interval size
    """
    DI = DecisionIntervals(test_data.env, None, 0.5, None)
    return DI.mean_alerts(test_data, DI.load_intervals(test_data, path, distrib), test_data.get_miniOD([]))


def beta_vs_size_err(test_data):
    """
    computes tests values for 25 different betas and alpha=0.5
    :param test_data: env and data
    :return: dataframe of errors
    """
    model = ModelStations(ud, 'svd', 'gbt', dim=7)
    # model.load()
    model.train(test_data)
    alpha = 0.5
    n = 25
    max = 0.99999999
    distrib = 'P'
    res = pd.DataFrame(index=np.linspace(0, max, n),
                       columns=['mean_lost_trips_arr', 'mean_lost_trips_dep', 'mean_interval_size',
                                'mean_interval_size_%'])
    features = get_features(ud)
    for beta in np.linspace(0, max, n):
        print(beta)
        DI = DecisionIntervals(ud, model, arr_vs_dep=alpha, beta=beta)
        DI.compute_min_max_data(features, test_data, **{'distrib': distrib})
        res.loc[beta, 'mean_lost_trips_arr'] = eval_worst_case(test_data, distrib,
                                                               test_data.env.decision_intervals,
                                                               arr_dep='arr')
        res.loc[beta, 'mean_lost_trips_dep'] = eval_worst_case(test_data, distrib,
                                                               test_data.env.decision_intervals,
                                                               arr_dep='dep')
        res.loc[beta, 'mean_interval_size'] = mean_interval_size(test_data, distrib,
                                                                 test_data.env.decision_intervals)
        res.loc[beta, 'mean_interval_size_%'] = mean_interval_size_percent(test_data, distrib,
                                                                           test_data.env.decision_intervals)
    print(res)
    return res

def compare_bixi_opt(d):
    """
    compare the repartition of rebalancing operations during the day
    :param d: data (Data object)
    :return: none
    """
    di = 'D:/maitrise/code/resultats/stations_bixi_min_max_target.csv'
    dis = ''
    mabixi = mean_alerts(d, dis, di)
    mod = ModelStations(d.env, 'svd', 'gbt', dim=10, **{'var': True})
    mod.load()
    DI = DecisionIntervals(d.env, mod, 0.5, 0.65)
    WH = mod.get_all_factors(d)
    dis = 'P'
    DI.compute_min_max_data(WH, d, True, **{'distrib': dis})
    di = d.env.decision_intervals
    ma = mean_alerts(d, dis, di)
    from matplotlib import cm
    cmap = cm.Paired
    plt.plot(ma[2], label='departures 0.5 0.65', color=cmap(1/12))
    plt.plot(ma[3], label='arrivals 0.5 0.65', color=cmap(3/12))
    plt.plot(mabixi[2], label='departures bixi', color=cmap(1/12), linestyle=':')
    plt.plot(mabixi[3], label='arrivals bixi', color=cmap(3/12), linestyle=':')
    plt.axvline(x=6, linestyle=':',color='k')
    plt.axvline(x=11, linestyle=':',color='k')
    plt.axvline(x=15, linestyle=':',color='k')
    plt.axvline(x=20, linestyle=':',color='k')
    plt.legend()
    plt.xlabel('Hour')
    plt.ylabel('Number of rebalancing per hour')
    plt.show()


def compute_scores(d, train, mod_name, alpha, beta):
    """
    compute the test scores
    :param d: data obj (test data)
    :param train: train data (Data obj)
    :param mod_name: 'bixi' to test bixi intervals, free for others (appear in the resulting dataframe)
    :param alpha: alpha parameter
    :param beta: beta parameter
    :return: score dataframe
    """
    if mod_name == 'bixi':
        di = 'D:/maitrise/code/resultats/stations_bixi_min_max_target.csv'
        dis = ''
    else:
        env = Environment('Bixi', 'train')
        # data = Data(env)
        mod = ModelStations(train.env, 'svd', 'gbt', dim=10, **{'var': True})
        # mod = CombinedModelStation(env, **{'var': True})
        # mod.train(train)
        # mod.save()
        mod.load()
        DI = DecisionIntervals(train.env, mod, alpha, beta)
        WH = mod.get_all_factors(d)
        dis = 'P'
        DI.compute_min_max_data(WH, d, True, **{'distrib': dis})
        di = d.env.decision_intervals
    df = pd.DataFrame(
        columns=['DI', 'alpha', 'beta', 'lost_arr', 'lost_dep', 'eval_target', 'mean_size', 'mean_size_%', 'sum_min',
                 'sum_max', 'sum_tar',
                 'mean_alerts_dep', 'mean_alerts_arr', ])
    ma = mean_alerts(d, dis, di)
    si = sum_int(d, dis, di)
    df.loc[0, :] = [mod_name,
                    alpha,
                    beta,
                    eval_worst_case(d, dis, di, arr_dep='arr'),
                    eval_worst_case(d, dis, di, arr_dep='dep'),
                    eval_target(d, dis, di),
                    mean_interval_size(d, dis, di),
                    mean_interval_size_percent(d, dis, di),
                    si[0],
                    si[1],
                    si[2],
                    ma[0],
                    ma[1],
                    ]
    print(df)

    return df


def compute_scores_def(d, train):
    """
    compute the scores for several values of alpha and beta and saves to  di_scores.csv file
    :param d: test data
    :param train: train data
    :return: none
    """
    df = compute_scores(d, train, 'bixi', None, None)
    l = []
    for b in np.linspace(0, 1, 21):
        l.append(compute_scores(d, train, '0.5_' + str(b), 0.5, b))
    df1 = pd.concat(l)

    l = []
    for a in np.linspace(0, 1, 21):
        l.append(compute_scores(d, train, str(a) + '_0.65', a, 0.65))
    df2 = pd.concat(l)
    df = pd.concat([df, df1, df2])
    print(df)
    df.to_csv('di_scores.csv')



if __name__ == '__main__':
    from preprocessing.Environment import Environment
    from preprocessing.Data import Data

    # ud = Environment('Bixi', 'train')
    # d = Data(ud)
    # alpha_vs_scores(d)
    # mod = ModelStations(ud, 'svd', 'gbt', dim=7)
    # mod = CombinedModelStation(ud)
    # mod.train(d)
    # mod.save()
    # mod.load()
    ud = Environment('Bixi', 'test')
    d = Data(ud)
    ud = Environment('Bixi', 'train')
    train = Data(ud)
    # DI = DecisionIntervals(ud, mod, 0.5, 0)
    # WH = mod.get_all_factors(d)
    # DI.compute_min_max_data(WH, d, True, **{'distrib': 'P'})
    compute_scores_def(d, train)
    # b = 0.65
    # a = 0.5
    # compute_scores(d, train, str(a)+'_' + str(b), a, b)
    # compute_scores(d, train, 'bixi', None, None)
    # compare_bixi_opt(d,train)
    # beta_vs_size_err(d)
