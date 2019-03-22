import matplotlib.pyplot as plt
from tensorflow.python.framework.errors_impl import InternalError
from model_station.Reduction import *
from utils.modelUtils import *
from preprocessing.Environment import Environment
import time

def SVD_analysis(data):
    """
    plots the loss in function of dimension of the reduction
    :param data: a data object with testing data
    :return: none
    """
    s = Reduction(data.env,hours=[])
    svd = TruncatedSVD(len(s.preselect) - 1)
    svd.fit(data.get_miniOD([],log=True)[s.preselect].as_matrix())
    t = np.zeros(svd.explained_variance_ratio_.shape[0] + 1)
    t[1:] = svd.explained_variance_ratio_
    plt.plot(np.cumsum(t))
    print((svd.explained_variance_ratio_))
    plt.show()
    svd = TruncatedSVD(50)
    svd.fit(data.get_miniOD([],log=True)[s.preselect].as_matrix())
    plt.plot(svd.explained_variance_ratio_)
    plt.show()


def SVD_generalisation_test(data):
    s = Reduction(data.env)
    train = data.get_partialdata_per(0, 0.8).get_miniOD([], log=False)[s.preselect].as_matrix()
    test = data.get_partialdata_per(0.8, 1).get_miniOD([], log=False)[s.preselect].as_matrix()
    err=[]
    for i in range(1,1000):
        svd = TruncatedSVD(i)
        svd.fit(train)
        p = svd.inverse_transform(svd.transform(test))
        e = rmse(test,p)
        err.append(e)
        print(e)
    plt.plot(err)
    plt.show()



def hierarchy_analysis(data):
    m = 'weighted'
    s = Reduction(data.env)
    learn = data.get_synthetic_miniOD([],False)[s.preselect].as_matrix()
    norm = learn.sum(axis=0)
    l = np.zeros((learn.shape[0] + 1, learn.shape[1]))
    l[0, :] = norm
    l[1:, :] = learn / norm
    d = 'seuclidean'
    d = ClusteringHierarchy.distance
    lin = linkage(
        l.T,
        metric=d,
        method=m
    )
    from scipy.cluster.hierarchy import dendrogram
    dendrogram(lin)
    plt.title('dendogram, ' + m + ' hierarchical clustering')
    plt.show()
    plot_clusters(data, fcluster(lin, 10, criterion='maxclust') - 1, title='cluster mean behavior ' + m + ' algorithm')


def plot_clusters(data, labels, title=''):
    data = data.get_partialdata_per(0, 0.5)
    nl = len(np.unique(labels))
    s = Reduction(data.env)
    d = data.get_synthetic_miniOD([],False)[s.preselect].as_matrix()
    t = np.zeros((d.shape[0], nl))
    r = np.zeros((24*7, nl))
    for l in range(nl):
        t[:, l] = d[:, labels == l].mean(axis=1)
    for h in range(24*7):
        r[(h+3*24-2) % (7*24), :] = t[range(h, t.shape[0], 24*7), :].mean(axis=0)
    styles = ['-','--','-','--','-','--','-','--']
    markers = ['+','+','.','.','x','x',',',',']
    plt.figure(figsize=(20,10))
    for p in range(8):
        # plt.subplot(2, 4, p+1)
        plt.plot(r[:, p], label='cluster ' + str(p),marker=markers[p], linestyle=styles[p])
        plt.legend()
    fig = plt.subplot(111)
    fig.set_ylabel('Centroid value')
    fig.set_xlabel('UTC timestamp')
    plt.title(title)
    plt.xticks(range(0,169,24))
    plt.savefig('kmeans_components2.pdf')
    plt.show()


def SVD_plot_dim(data):
    s = Reduction(data.env)
    svd = TruncatedSVD(8)
    t = svd.fit_transform(data.get_miniOD([])[s.preselect].as_matrix())
    t = t[:100*24,:]
    r = np.zeros((24*7, 8))
    translate = 24*3
    for h in range(24*7):
        r[(translate+h)%168, :] = t[range(h, t.shape[0], 24*7), :].mean(axis=0)
    styles = ['-','-','-','--',':']
    markers = ['+','.','x',',',',']
    plt.figure(figsize=(8,4))
    for p in range(5):
        # plt.subplot(2, 4, p+1)
        plt.plot(r[:, p], label='component ' + str(p),
                 # marker=markers[p],
                 linestyle=styles[p])
        plt.legend()
    fig = plt.subplot(111)
    fig.set_ylabel('Mean value of component')
    fig.set_xlabel('Hour')
    plt.xticks(range(0,169,24))
    plt.savefig('SVD_components2.pdf', bbox_inches='tight')
    plt.show()


def redution_plot_dim(data, red, dim=10):
    s = get_reduction(red)(data.env, dim=dim)
    s.train(data=data)
    t = s.transform(data)
    # t = svd.fit_transform(data.get_miniOD()[s.preselect].as_matrix())
    r = np.zeros((24, t.shape[1]))
    for h in range(24):
        r[h, :] = t[range(h, t.shape[0], 24), :].mean(axis=0)
    for p in range(t.shape[1]):
        # plt.subplot(2, 4, p+1)
        plt.plot(r[:, p], label='component ' + str(p))
        plt.legend()
    fig = plt.subplot(111)
    plt.title(red)
    fig.set_ylabel('mean value of component')
    fig.set_xlabel('UTC timestamp')
    plt.show()


def PCA_analysis(data):
    """
    plots the loss in function of dimension of the reduction
    :param data: a data object with testing data
    :return: none
    """
    s = Reduction(data.env,hours=[])
    svd = PCA(len(s.preselect) - 1)
    svd.fit(data.get_miniOD([])[s.preselect].as_matrix())
    plt.plot(svd.explained_variance_ratio_)
    plt.show()
    svd = PCA(50)
    svd.fit(data.get_miniOD([])[s.preselect].as_matrix())
    plt.plot(svd.explained_variance_ratio_)
    plt.show()


def kmeans_analysis(data):
    """
    plots the loss in function of the number of clusters
    :param data: a data object with testing data
    :return: none
    """
    s = Reduction(data.env)
    l = []
    for k in range(500):
        print(k)
        kmeans = KMeans(k + 1)
        kmeans.fit(data.get_miniOD([])[s.preselect].as_matrix())
        l.append(kmeans.inertia_)
        if k % 10 == 0:
            plt.plot(l)
            plt.show()

def kmeans_svd_analysis(data):
    """
    plots the loss in function of the number of clusters
    :param data: a data object with testing data
    :return: none
    """
    s = Reduction(data.env)
    Kmeans = get_reduction('kmeans')
    SVD = get_reduction('svd')
    l_km= []
    l_svd = []
    for k in range(500):
        kmeans = Kmeans(data.env,dim=k+1)
        print(k+1)
        kmeans.train(data)
        r = kmeans.inv_transform(kmeans.transform(data))
        err = ((data.get_miniOD([])[s.preselect].as_matrix()-r)**2).mean()
        l_km.append(err)
        svd = SVD(data.env,dim=k+1)
        svd.train(data)
        r = svd.inv_transform(svd.transform(data))
        err = ((data.get_miniOD([])[s.preselect].as_matrix()-r)**2).mean()
        l_svd.append(err)

        if k==30:
            plt.plot(l_km, label='kmeans')
            plt.plot(l_svd, label='svd')
            plt.xlabel('Dimensions (k)')
            plt.ylabel('RL')
            plt.legend()
            plt.savefig('kmeans_svd.pdf', bbox_inches='tight')
            plt.show()



def kmeans_print_clusters(data):
    s = Reduction(data.env)
    kmeans = KMeans(10)
    kmeans.fit(data.get_miniOD([])[s.preselect].as_matrix().T)
    print(kmeans.labels_.shape)
    plot_clusters(data, kmeans.labels_ - 1)


def autoencoder_analysis(data):
    s = Reduction(data.env)
    train = data.get_partialdata_per(0, 0.8)
    test = data.get_partialdata_per(0.8, 1)
    M = (s.get_y(train.get_miniOD([])).as_matrix() ** 2).sum()
    print(M)
    param = {
        'std_noise': 0.2,
        'activation': 'relu',
        'noeuds': [ 70, 35],
        'dropout': 1,
        'learning_rate': 2,
        # 'noeuds': [250, 125, 70, 35, self.dim],

        'epsilon': 6.8129e-05,
        # 'learning_rate': 0.01,
        'reg_l1': 0,
        'reg_l2': 0,
        'beta_1': 0.977,
        'beta_2': 0.99973,
        'decay': 0,
        # 'n_couches': 5
        'n_epochs': 100,
        'batch_size': 30,
        'verb': 2
    }
    auto = DimRedAutoEncoder(data.env,dim=5,**param)
    auto.train(train, **param)
    auto.save(add_path=data.env.system)
    x = auto.inv_transform(auto.transform(test))
    err= ((x - s.get_y(test.get_miniOD([]))) ** 2).as_matrix().sum()
    print(err)
    print((M-err)/M)

    return

def compute_info_loss(data):
    """
    compute the sum of errors made by redution algorithms
    :param data: a Data object with the testing data 
    :return: 
    """
    dims = {
        'svd': 5,
        'pca': 5,
        'autoencoder': 5,
        'kpca': 5,
        'kmeans': 5,
        'average': 10,
        'complete': 10,
        'weighted': 10,
        'GM': 5,
        'SC': 10,
        'maxkcut': 10,
        'ODkmeans': 10,
    }
    train = data.get_partialdata_per(0, 0.8)
    test=train

    # test = data.get_partialdata_per(0.8,1)
    s = Reduction(train.env)
    D = s.get_y(train.get_miniOD([])).as_matrix()
    M =((D-D.mean()) ** 2).mean()
    errr = {}
    nnn = algorithms.keys()
    nnn= ['kmeans','svd','autoencoder','GM','sum']
    temps = {}
    for n in nnn:
        print(n)
        try:
            print(dims[n])
            red = get_reduction(n)(train.env, dim=dims[n])
        except KeyError:
            red = get_reduction(n)(train.env, dim=10)
        # try:
        #     red.load(train.env.system)
        # except (FileNotFoundError, OSError,InternalError):
        st = time.time()
        red.train(train)
        temps[n]=time.time()-st
            # red.save(train.env.system)
        x = type(red).inv_transform(red,type(red).transform(red,test))
        if n=='kmeans':
            print(n, x.shape)
        err = ((x - s.get_y(test.get_miniOD([]))) ** 2).as_matrix().mean()
        errr[n] = err
    print(M)
    for n in nnn:
        print(n, errr[n], 1-errr[n]/M, temps[n])


if __name__ == '__main__':
    ud = Environment('Bixi','train')
    d = Data(ud)
    # kmeans_svd_analysis(d)
    import tensorflow as tf
    with tf.device('/cpu:0'):
        kmeans_svd_analysis(d)
        # autoencoder_analysis(d)
        # kmeans_print_clusters(d)
    # kmeans_print_clusters(d)
    # hierarchy_analysis(d)
    # redution_plot_dim(d,'SC', dim=10)
