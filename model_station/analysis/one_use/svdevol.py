simport pandas as pd
import matplotlib.pyplot as plt
from config import root_path
import numpy as np
from preprocessing.Data import Data
from model_station.ModelStations import ModelStations

def scores():
    m10 = pd.read_csv(root_path+'svdevol.csv')
    m3 = pd.read_csv(root_path+'svdevol3.csv')
    m5 = pd.read_csv(root_path+'svdevol5.csv')
    svd={'rmsle':0.372,'mape':0.25,'r2':0.339,'rmse':0.996}
    for n,i in enumerate(['rmsle','mape','r2','rmse']):
        plt.subplot(2,2,n+1)
        plt.title(i)
        plt.plot(m3['weeks'],m3[i],label='3 components')
        plt.plot(m5['weeks'],m5[i],label='5 components')
        plt.plot(m10['weeks'],m10[i],label='10 components')
        plt.plot((0,10),(svd[i],svd[i]),label='svd')
        plt.legend()
    plt.show()

def week():
    train = Data(second_name='train')
    test = Data(second_name='test')


    h=168
    train2=train.get_partialdata_n(-h,-1)
    test = test.get_partialdata_n(0,168)

    mod1=ModelStations(train.env,'svd','gbt')
    # mod1.train(train)
    # mod1.save()
    mod2=ModelStations(train.env,'svdevol','gbt',dim=3,**{'n_week':h/168})
    # mod2.train(train)
    # mod2.train_inv(train2)
    # mod2.save()
    mod3=ModelStations(train.env,'svdevol','gbt',dim=5,**{'n_week':h/168})
    # mod3.train(train)
    # mod3.train_inv(train2)
    # mod3.save()
    mod4=ModelStations(train.env,'svdevol','gbt',dim=10,**{'n_week':h/168})
    # mod4.train(train)
    # mod4.train_inv(train2)
    # mod4.save()

    mod1.load()
    mod2.load()
    mod3.load()
    mod4.load()

    p1 = mod1.predict(test)
    p2 = mod2.predict(test)
    p3 = mod3.predict(test)
    p4 = mod4.predict(test)
    real = test.get_miniOD([])[test.get_stations_col()].to_numpy()

    while True:
        s=np.random.randint(0,900)
        plt.plot(p1[:,s], label='svd')
        plt.plot(p2[:,s], label='svdevol3')
        plt.plot(p3[:,s], label='svdevol5')
        plt.plot(p4[:,s], label='svdevol10')
        plt.plot(real[:,s], label='real')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    scores()