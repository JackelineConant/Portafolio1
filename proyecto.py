import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Algoritmo para la regresión lineal
#--------------------------------------------------------------------------------------------------------
# Función que va a ir aumentando el peso (w) y el bias durante una iteracion (epoch)
def update_w_and_b(X, y, w, b, alpha):
    #Actualizan los parámetros w y b durante una iteración 
    N = len(y)
    y_pred = np.dot(X, w) + b
    error = y - y_pred

    # Gradiantes de peso y bias ------------------------------------------------------------------------- 
    dl_dw = -2 * np.dot(X.T, error) / N
    dl_db = -2 * np.sum(error) / N

    # Actualiza w y b y lo retornan ---------------------------------------------------------------------
    w -= alpha * dl_dw
    b -= alpha * dl_db
    return w, b

# Función para poder calcular la función de Error Cuadrático Medio(MSE) ---------------------------------
def avg_loss(X, y, w, b):
    #Calculan el error y lo retornan
    N = len(y)
    y_pred = np.dot(X, w) + b
    total_error = np.sum((y - y_pred) ** 2)
    return total_error / N

# Función para el entrenamiendo del modelo de regresión logística ---------------------------------------
def train(X, y, w, b, alpha, epochs):
    #Hacen un bucle que se loopea sobre multiples iteraciones y al final imprime el progreso
    print('Progreso de entrenamiento:')
    for e in range(epochs):
        w, b = update_w_and_b(X, y, w, b, alpha)
        if e % 400 == 0 or e == epochs - 1:
            avg_loss_ = avg_loss(X, y, w, b)
            print(f"Epoch {e} | Loss: {avg_loss_:.4f} | Weights: {w} | Bias: {b:.4f}")
    return w, b

# Función para predecir valores -------------------------------------------------------------------------
def predict(X, w, b):
    #Hace predicciones en base a los datos de entrenamiento
    return np.dot(X, w) + b

#--------------------------------------------------------------------------------------------------------

# Cargar la base de datos a utilizar----------------------------------------------------------------------
# En este caso yo utilicé la base de datos de Real Estate Price(Precio Inmoviliario)
# https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction
cdata = pd.read_csv('Real estate.csv')
# Visualizar tabla (Transpose data)
print('Datos Real estate Price T\n')
print(cdata.head().T)
# Visualizar su información
print('Información sobre el dataset\n')
print(cdata.info())

# Eliminar los valores que no son de importancia para la regresión lineal
cdata.drop(columns=['No','X1 transaction date'], inplace=True)
print('Base de datos sin las variables No y X1 transaction date\n')
print(cdata.head())

# Hace la función shuffle para desordenar los datos del dataset
shuffle_df = cdata.sample(frac=1).reset_index(drop=True)
X_original = shuffle_df.drop(columns=['Y house price of unit area'])
y = shuffle_df['Y house price of unit area']
# Visualizar las columnas de X y Y. 
# Donde X son todos los datos sobrantes excepto 'Y house price of unit area', ya que esa será nuestra label(y)
print('Valores de features x están mezcladas:\n')
print(X_original)
print('Valores del label y está mezclada:\n')
print(y)

# Normaliza los datos utilizando el escalador de datos--------------------------------------------------------
# El propósito de normalizar los datos es tratar de hacerlos más cercanos y evitar puntos de datos muy 
#   altos que puedan afectar las futuras predicciones del modelo entrenado.
dataScaler = StandardScaler()
scaler = dataScaler.fit(X_original)
X_scaled = scaler.transform(X_original)
# Muestra el arreglo normalizado resultante
print("Valores de features standarizados:\n")
print(X_scaled)

# Crea un dataframe con los datos normalizados
X = pd.DataFrame(X_scaled)
X.columns = X_original.columns
# Muestra las primeras 5 líneas del dataframe resultante
print(X.head())

# Hacer el split de los datos de entrenamiento(train set)80% y evaluación(test set)20%
train_size = int(0.8 * len(cdata))

X_train = X[:train_size]
Y_train = y[:train_size]
X_test = X[train_size:]
Y_test = y[train_size:]

# Inicializar el peso y bias
w = np.zeros(X_train.shape[1])
b = 0.0

# Hyperparametros inicializados
alpha = 0.001  # Learning rate
epochs = 12000  # Number of epochs

# Entrenar el modelo con TODOS los features X de entrenamiento
w, b = train(X_train, Y_train, w, b, alpha, epochs)

# Evaluar el modelo con el Test set
Y_pred = predict(X_test, w, b)

# Calcula y imprime el error final en el Test Set
test_loss = avg_loss(X_test, Y_test, w, b)
print(f"Final Test Loss: {test_loss:.4f}")

#---------------------------------------------------------------------------------------------------

# Seleccionar el valor a utilizar en X para la regresión lineal SOLO UN feature X de entrenamiento 
X_train = X_train['X2 house age']
X_test = X_test['X2 house age']
# Muestra los datos de un solo feature de entrenamiento, que en este caso es la edad de la casa :(
print("Valores de un solo Feature X seleccionado")
print(X_train)

#Volver a iniciar los datos para los nuevos valores de X -------------------------------------------
# Inicializar el peso y bias
w = 0.0
b = 0.0
# Hyperparametros
alpha = 0.001  # Learning rate(alpha)
epochs = 12000  # Número de epochs(iteraciones)

# Entrenar al modelo de regresión lineal
w, b = train(X_train, Y_train, w, b, alpha, epochs)
# Evaluar el modelo con el Test set
Y_pred = predict(X_test, w, b)

# Calcula y imprime el error final en el Test Set
test_loss = avg_loss(X_test, Y_test, w, b)
print(f"Final Test Loss: {test_loss:.4f}")

# Visualiza los resultados de la gráfica de regresión lineal de un solo feature
plt.scatter(X_test[:], Y_test, color='blue')  # Graficando feature(X2 house age) vs. target(Y house price of unit area)
plt.plot(X_test[:], predict(X_test, w, b), color='red')
plt.xlabel('X2 house age')
plt.ylabel('Y house price of unit area')
plt.title('Linear Regression on Test Set')
plt.show()

#Función para mostrar el progreso del modelo de regresión lineal de UN SOLO feature X y el label Y 
def train_and_plot(X, y, w, b, alpha, epochs):
    #Bucle que hace loop sobre multiples epochs(iteraciones) y muestra el progreso de la gráfica
    for e in range(epochs):
        w, b = update_w_and_b(X, y, w, b, alpha)
        # Graficar visualmente la última iteración
        if e == epochs-1:
            avg_loss_ = avg_loss(X, y, w, b)
            plt.scatter(X[:], y, color='blue')  # Plotting first feature vs. target
            plt.plot(X[:], np.dot(X, w) + b, color='red')
            plt.title("Epoch {} | Loss: {} | w:{}, b:{}".format(e, round(avg_loss_,2), round(w, 4), round(b, 4)))
            plt.show()
    return w, b
    
#Para esta parte fue necesario editar los epochs para que se dividieran por su total y así fueran los datos
#   pudieran ser captados mejor dentro de la gráfica.
epoch_plots = [1, int(epochs/50), 2*int(epochs/50), 3*int(epochs/50), 4*int(epochs/50), epochs+1]
for epoch_plt in epoch_plots:
    w, b = train_and_plot(X_train, Y_train, 0.0, 0.0, alpha, epoch_plt)

#---------------------------------------------------------------------------------------------------------
#Comentarios:
'''
El modelo de regresión lineal creado puede entrenarse con todos los predictores. Sin embargo, para la 
visualización gráfica, se decidió utilizar solo una característica (feature) para mantener la gráfica más 
limpia y evitar confusiones.

Como reflexión, podemos ver que la regresión lineal es una técnica fundamental para analizar datos, 
especialmente en el campo del machine learning. Su objetivo es modelar la relación entre una variable 
dependiente (label) y una o más variables independientes (features). En las últimas funciones de este 
código, exploramos una de sus formas más simples, donde buscamos que el modelo de regresión lineal 
encontrara la línea recta que mejor se ajustara a nuestros datos.

Una de las cosas que más me interesaron de este proyecto fue observar cómo, al agregar más características 
al modelo, el error final o costo se reducía significativamente, mucho más que cuando se utilizaba solo un 
predictor como variable x.
'''

#---------------------------------------------------------------------------------------------------------