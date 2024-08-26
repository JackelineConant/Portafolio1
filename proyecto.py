#Importar librerías
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Importar algoritmo de regresión 
from regression import *

#Inicializar la base de datos a utilizar
lol = pd.read_csv('global_population_growth_2024.csv')
print(lol.head())
#lol.plot()
lol.plot(kind = 'scatter', x = 'Attack damage per lvl', y = 'AS ratio')
plt.show()