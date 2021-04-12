# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 14:27:07 2020

@author: Condominios Manzano
"""
 #PCA con Sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as sc
import numpy as np
filepath="C:/Users/Condominios Manzano/Desktop/Machine Learning/python-ml-course-master/python-ml-course-master/datasets/iris/iris.csv"
df=pd.read_csv(filepath)
Y=df.iloc[:,-1]
X=df.iloc[:,:-1]
X_std=sc().fit_transform(X)

pca=PCA(n_components=2)
y=pca.fit_transform(X_std)
