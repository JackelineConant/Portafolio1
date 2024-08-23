#Importar librerías
import pandas as pd
import matplotlib.pyplot as plt

#Inicializar la base de datos a utilizar
lol = pd.read_csv('lol_champ.csv')
print(lol.head())
lol.plot()
lol.plot(kind = 'scatter', x = 'Attack damage per lvl', y = 'AS ratio')

plt.show()
#print(plot)

