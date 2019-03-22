from decision_intervals.decision_intervals import *


class MovingDecisionInterval(DecisionIntervals):
    def eval_moving(self, test_data, distrib='NB'):
        hparam = {
            'distrib': distrib
        }
        df = test_data.get_miniOD()
        df = df.reset_index()
        worst_case, target, size = [], [], []
        for i in range(0, df.shape[0] - 10, 7 * 24):
            WT = df[i:].reset_index()
            intervals = self.compute_min_max_data(WT, test_data, save=False, **hparam)
            worst_case.append(self.eval_worst_case(test_data, distrib, intervals, WT, single=True))
            target.append(self.eval_target(test_data, distrib, intervals, WT, single=True))
            size.append(self.mean_interval_size(test_data, distrib, intervals))
        worst_case = np.array(worst_case)
        target = np.array(target)
        size = np.array(size)
        return worst_case.mean(),target.mean(),size.mean()


if __name__ == '__main__':
    env = Environment('Bixi', 'train')
    data = Data(env)
    mod = ModelStations(env, 'svd', 'gbt', dim=5)
    mod.load()
    DI = MovingDecisionInterval(env, mod, 1 / 2, 0.9)
    WH = mod.get_all_factors(data).loc[0:, :]
    DI.general_min_max(WH, data, True, **{'distrib': 'ZI'})
    DI.general_min_max(WH, data, True, **{'distrib': 'P'})
    DI.general_min_max(WH, data, True, **{'distrib': 'NB'})