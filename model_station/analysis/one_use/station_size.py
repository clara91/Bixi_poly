import pandas as pd
import numpy as np
from utils.modelUtils import maxi
from config import root_path
import matplotlib.pyplot as plt

csv = pd.read_csv(root_path+'stations_scores.csv')
score = 'r2'
plt.figure(figsize=(15,7))
plt.scatter(csv['size'],maxi(0,csv[score]))
plt.xlabel('Station Sizes')
plt.ylabel(score.upper())
plt.savefig('station_size_'+score+'.pdf',bbox_inches='tight')
plt.show()

lcb=[]
for i in np.linspace(0, np.max(csv['size'])):
    lcb.append((csv['size'] <= i).mean())
plt.plot(np.linspace(0, np.max(csv['size'])), lcb)
plt.show()
print((csv['size']<0.3).mean())
print((csv['size']>2).mean())