import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import root_path

def sept():
    score='r2'
    d = pd.read_csv(root_path+'time_boostingsept6.csv')
    cols = [score+'',score+'_base',score+'.1',score+'.2',score+'.3']
    d = d[cols]
    d.drop(index=1,inplace=True)
    d.drop(index=0,inplace=True)
    d.reset_index()
    decalage=6
    d['Jour'] = pd.Series(d.index.values).apply(lambda x:int((x+decalage)/24))
    d['Heure'] = pd.Series(d.index.values).apply(lambda x:(x+decalage)%24)
    d['leg']= pd.Series(d.index.values).apply(lambda x: str(int((x+decalage)/24))+ ' ' +str((x+decalage)%24))
    d['seg']= pd.Series(d.index.values).apply(lambda x:int((x+decalage-3)/24))
    for c in cols:
        if c!='leg':
            d[c]=pd.to_numeric(d[c])
    d=d[(d['Heure']>13)+(d['Heure']<3)]
    d['x']=range(d.shape[0])
    colors = ['b','k','r','g']
    plt.figure(figsize=[15,5])
    for j in np.unique(d['seg']):
        l1, = plt.plot(d['x'][d['seg']==j], d[score+''][d['seg']==j], c=colors[0], label='model 1')
        l3, = plt.plot(d['x'][d['seg']==j], d[score+'.1'][d['seg']==j], c=colors[2], label='model 2')
        # l4, = plt.plot(d['x'][d['seg']==j], d[score+'.2'][d['seg']==j], c=colors[3], label='model 3')
        l2, = plt.plot(d['x'][d['seg']==j], d[score+'_base'][d['seg']==j], c=colors[1], label='base model')
    plt.legend(handles=[l2,l1,l3])
    plt.xlabel('day and hour')
    plt.ylabel(score+' score')
    plt.xticks(d['x'],d['leg'], rotation='vertical')
    plt.savefig('sept_error'+score+'.pdf',bbox_inches='tight')
    plt.show()
sept()


def jazz():
    score = 'r2'
    d = pd.read_csv(root_path+'time_boostingjazz6.csv')
    # d.drop(index=1,inplace=True)
    # d.drop(index=0,inplace=True)
    d.reset_index()
    decalage=6
    d['Jour'] = pd.Series(d.index.values).apply(lambda x:int((x+decalage)/24))
    d['Heure'] = pd.Series(d.index.values).apply(lambda x:(x+decalage)%24)
    d['leg']= pd.Series(d.index.values).apply(lambda x: str(int((x+decalage)/24))+ ' ' +str((x+decalage)%24))
    d['seg']= pd.Series(d.index.values).apply(lambda x:int((x+decalage-3)/24))

    cols = [score+'',score+'_base',score+'.1',score+'.2',score+'.3','Heure','Jour','leg','seg']
    d = d[cols]
    for c in cols:
        if c!='leg':
            d[c]=pd.to_numeric(d[c])
    d=d[(d['Heure']>13)+(d['Heure']<3)]

    d['x']=range(d.shape[0])
    colors = ['b','k','r','g']
    plt.figure(figsize=[15,5])
    for j in np.unique(d['seg']):
        l1, = plt.plot(d['x'][d['seg']==j], d[score+''][d['seg']==j], colors[0], label='model 1')
        l3, = plt.plot(d['x'][d['seg']==j], d[score+'.1'][d['seg']==j], colors[2], label='model 2')
        # l4, = plt.plot(d['x'][d['seg']==j], d[score+'.2'][d['seg']==j], colors[3], label='model 3')
        l2, = plt.plot(d['x'][d['seg']==j], d[score+'_base'][d['seg']==j], colors[1], label='base model')
    plt.legend(handles=[l2,l1,l3])
    plt.xlabel('day and hour')
    plt.ylabel(score+' score')
    plt.xticks(d['x'],d['leg'], rotation='vertical')
    plt.savefig('jazz_error'+score+'.pdf',bbox_inches='tight')
    plt.show()
jazz()