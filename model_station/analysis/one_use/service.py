import numpy as np
import matplotlib.pyplot as plt
from config import root_path
from preprocessing.Data import Data
m = np.loadtxt(root_path+'test.csv')

d = Data(second_name='train')
l=[5,10,11,15,102,506]

l=[33,35,62,87]
# l=range(m.shape[1])
plt.figure(figsize=(10,5))
for i in l:
    plt.plot(m[:,i][m[:,i]!=0],label=d.get_stations_col(None)[i].replace('date ',''))
    # plt.title(i)
    # plt.show()
plt.legend()
plt.xlabel('Station Inventory')
plt.ylabel('Service level')
plt.savefig('service_levels.pdf',bbox_inches='tight')
plt.show()