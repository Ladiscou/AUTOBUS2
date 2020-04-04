import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.preprocessing import normalize
#Imports pour les scores
from libscores import get_metric
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

limd2012 = [10,12,2,4,7]
lijd2012 = [5,5,5,5,3]
limr2012 = [9,11,1,2,4]
lijr2012 = [1,3,1,6,6]
limd2013 = [2,4,7,10,12]
lijd2013 = [5,5,3,5,5]
limr2013 = [1,3,5,7,11]
lijr2013 = [6,6,6,1,6]
limd2014 = [3,4,7,10,12]
lijd2014 = [5,5,5,5,5]
limr2014 = [1,3,5,9,11]
lijr2014 = [6,6,6,6,6]
limd2015 = [2,4,7,10,12]
lijd2015 = [5,5,5,5,5]
limr2015 = [1,2,4,9,11]
lijr2015 = [6,6,6,1,6]
limd2016 = [2,4,7,10,12]
lijd2016 = [5,5,1,2,5]
limr2016 = [1,2,5,9,11]
lijr2016 = [6,6,6,3,3]
limd2017 = [2,4,7,10,12]
lijd2017 = [5,2,5,5,5]
limr2017 = [1,3,5,9,11]
lijr2017 = [1,0,1,0,0]
limd2018 = [2,4,7,10,12]
lijd2018 = [5,5,5,5,5]
limr2018 = [1,2,4,7,11]
lijr2018 = [0,0,0,0,0]


def departs(datamodif):
    dep = np.zeros(shape = 38563)
    lij = np.zeros(shape = 5)
    lim = np.zeros(shape = 5)
    for d in range(datamodif.iloc[:,1].size):
        if(datamodif['year'][d] == 2012):
            lij = lijd2012
            lim = limd2012
        if(datamodif['year'][d] == 2013):
            lij = lijd2013
            lim = limd2013
        if(datamodif['year'][d] == 2014):
            lij = lijd2014
            lim = limd2014
        if(datamodif['year'][d] == 2015):
            lij = lijd2015
            lim = limd2015
        if(datamodif['year'][d] == 2016):
            lij = lijd2016
            lim = limd2016
        if(datamodif['year'][d] == 2017):
            lij = lijd2017
            lim = limd2017
        if(datamodif['year'][d] == 2018):
            lij = lijd2018
            lim = limd2018
        for j in range(len(lij)):
            jour = (datamodif['weekday'][d] == lij[j]).any() or (datamodif['weekday'][d] == ((lij[j]+1)%7)).any()
            mois = (datamodif['month'][d] == lim[j]).any()
            jour_mois =  jour.any() & mois.any()
            vacances = datamodif['holiday'][d] == 1
            if(jour_mois.any() and vacances.any()):
                dep[d] = 1
    datamodif.insert(59,'departure', dep, True)

def retour(datamodif):
    ret = np.zeros(shape = 38563)
    lij = np.zeros(shape = 5)
    lim = np.zeros(shape = 5)
    for d in range(datamodif.iloc[:,1].size):
        if(datamodif['year'][d] == 2012):
            lij = lijr2012
            lim = limr2012
        if(datamodif['year'][d] == 2013):
            lij = lijr2013
            lim = limr2013
        if(datamodif['year'][d] == 2014):
            lij = lijr2014
            lim = limr2014
        if(datamodif['year'][d] == 2015):
            lij = lijr2015
            lim = limr2015
        if(datamodif['year'][d] == 2016):
            lij = lijr2016
            lim = limr2016
        if(datamodif['year'][d] == 2017):
            lij = lijr2017
            lim = limr2017
        if(datamodif['year'][d] == 2018):
            lij = lijr2018
            lim = limr2018
        for j in range(len(lij)):
            jour = (datamodif['weekday'][d] == lij[j]).any() or (datamodif['weekday'][d] == ((lij[j]-1)%7)).any()
            mois = (datamodif['month'][d] == lim[j]).any()
            jour_mois =  jour.any() & mois.any()
            vacances = datamodif['holiday'][d] == 1
            if(jour_mois.any() and vacances.any()):
                ret[d] = 1;
    datamodif.insert(59,'return', ret, True)

def heure_pointe(datamodif):
    pointe = np.zeros(shape = 38563)
    for i in range(38563):
        if((7<=datamodif['hour'][i]<=9 or 16<=datamodif['hour'][i]<=19) and (datamodif['weekday'][i]!=5 and datamodif['weekday'][i]!=6)):
            pointe[i] = 1
    pointe = pd.DataFrame(pointe)
    datamodif.insert(59,'crowded', pointe, True)
    return pointe

def lightFilter(data):
    comp = 0
    data1, data2, index1, index2 =  train_test_split(data, index, train_size = 0.5)
    data3, data4, index3, index4 =  train_test_split(data1, index1, train_size = 0.5)
    data5, data6, index5, index6 =  train_test_split(data2, index2, train_size = 0.5)
    clf = LocalOutlierFactor(n_neighbors=int(data3.iloc[:,1].size/2), contamination = 0.08)
    inlier1 = clf.fit_predict(data3)
    inlier2 = clf.fit_predict(data4)
    inlier3 = clf.fit_predict(data5)
    inlier4 = clf.fit_predict(data6)
    i = 0
    outlier = []
    max = inlier1.size
    while(i< max):
        if(inlier1[i] == -1):
            outlier.append(index3[i])
            comp+=1
        if(inlier2[i] == -1):
            outlier.append(index4[i])
            comp+=1
        if(inlier3[i] == -1):
            outlier.append(index5[i])
            comp+=1
        if(inlier4[i] == -1):
            outlier.append(index6[i])
            comp+=1
        i+= 1
    print(comp)
    print(i)
    return outlier

def majData(data, outiler):
    for i in outiler:
        data = data.drop(i)
    return data