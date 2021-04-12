# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:32:19 2020

@author: Condominios Manzano
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as sc
import numpy as np #Libreria númerica
import pandas as pd

filepath="C:/Users/Condominios Manzano/Desktop/Machine Learning/python-ml-course-master/python-ml-course-master/datasets/iris/iris.csv"
df=pd.read_csv(filepath)
Y=df.iloc[:,-1]
X=df.iloc[:,:-1]
#Modelizar PCA desde 0
#3 metodos para encontrar los vectores propios y los valores propios
#1) Matriz de variazas y covarianzas
#2) Matriz de correlaciones
#3) metodo singular value descomposition

#Para el metodo 1 se normaliza ya que los valores estan en diferentes unidades

X_std=sc().fit_transform(X)
x_=np.mean(X_std, axis=0)
#1) Matriz de variazas y covarianzas
cov_= (1/(len(X)-1)) * ((X_std - x_).T @ (X_std-x_))
ceig_val, ceig_vec=np.linalg.eig(cov_)

#2) Matriz de correlaciones
corr_=np.corrcoef(X_std.T)
eig_val, eig_vec=np.linalg.eig(corr_)
cor=np.corrcoef(X.T)

#3) metodo singular value descomposition
u,s,v=np.linalg.svd(X_std.T)

# Seleccion de los componentes principales: se toman los vectores que expliquen la mayor parte de los datos
sum_com=ceig_val.sum()
pct_com=[i/sum_com for i in ceig_val]
pct_cum=np.cumsum(pct_com)


#Proyectar las variables en el nuevo espacio vectorial: se redce de 4 a 2 el espacio vectorial
#Y=X @ W donde W seran los vectores seleccionados que explican la mayor parte de datos
W=eig_vec[:,0:2]
Y_trans=X_std @ W
