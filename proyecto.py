#Importar librerías
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Importar algoritmo de regresión 
from regression import *

#Inicializar la base de datos a utilizar
cdata = pd.read_csv('cancer_data.csv')

#Damos un vistazo a como se ve la base de datos
print(cdata.head())

#Posteriormente limpiamos los datos 
#Borrar datos que no sean importante para las predicciones
cdata.drop(["id", "Unnamed: 32",],axis=1,inplace = True)
#Remplazar los datos de diagnosis, que utilizaremos para hacer nuestra regresión logística
cdata['diagnosis'].replace(['B', 'M'],[0, 1], inplace=True)

#Función de normalización 

#Dividir mis variables y(labels) y x(features)
y = cdata['diagnosis'].values
#--------------------------------
X = cdata.drop(['diagnosis'], axis=1)

#Analizar cada uno de los valores
'''
print(X)
print(y)
'''

#Dividir los datos en train y test

#Primeramente llamaremos una función shuffle para sortear los datos del training y test set
cdata = cdata.sample(frac=1)
#print(cdata)

#Posteriormente necesitaremos divir el dataset en un porcentaje,
#   por lo que haré uso de una variable ratio
ratio = 0.80
#En base a previos conocimientos, se me comentó que normalmente train set contiene el 80% de dataset
#   y test set contiene el 20% del dataset para mejor entrenamiento y predicciones
total_rows = cdata.shape[0]
train_size = int(total_rows*ratio) #basicamente se creo una función que tiene el 80% de las filas del dataset

#Finalmente se dividen los datos
train = cdata[0:train_size]
test = cdata[train_size:]

#Volvemos a dividirlos
#Train & Test X values
X_train = train.drop(['diagnosis'], axis=1)
#--------------------------------
X_test = test.drop(['diagnosis'], axis=1)
#Train & Test Y values
y_train = train['diagnosis'].values
#--------------------------------
y_test = test['diagnosis'].values

print(X_train)
print(X_test)

#Escalamos los datos con la función sklearn
'''sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  # Train model
  epochs = 1000
  alpha = 1
  best_params = log_regression4(X_train, y_train, alpha, epochs)'''