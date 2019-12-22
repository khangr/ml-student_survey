# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:34:38 2019

@author: okhangur
"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier  

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn.svm import SVC

dosya = pd.read_csv('C:/turkiye-student-evaluation_generic.csv')

giris_verileri = dosya.iloc[:, 
[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,
19,20,21,22,23,24,25,26,27,28,29,30,31,32]]

cikis = dosya.iloc[:,7]

basari=list()

kSayisi = 5

kf = KFold(n_splits=kSayisi)   

fSkor = list()



for i in range(10,201,10):
    
    siniflandir = RandomForestClassifier(n_estimators=i,random_state=0)
    
    toplamBasari = 0;

    toplamfSkor = 0;

    for egitim_index, test_index in kf.split(giris_verileri):

        scaler = preprocessing.StandardScaler()

        stdGiris = scaler.fit_transform(giris_verileri.iloc[egitim_index,:])

        stdTest = scaler.transform(giris_verileri.iloc[test_index,:])        

        siniflandir.fit(stdGiris, cikis[egitim_index] )        

        cikis_tahmin = siniflandir.predict(stdTest)

        toplamBasari += (accuracy_score(cikis[test_index], cikis_tahmin))        

        toplamfSkor += ( f1_score(cikis[test_index], cikis_tahmin, labels=None, pos_label=1, average='macro', sample_weight=None))        

    basari.append(toplamBasari/kSayisi)

    fSkor.append(toplamfSkor/kSayisi)










