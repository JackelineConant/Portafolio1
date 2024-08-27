import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Algoritmo para la regresión lineal
#--------------------------------------------------------------------------------------------------------
#Función que va a ir aumentando el peso (w) y el bias durante una iteracion (epoch)
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

#Cargar la base de datos a utilizar----------------------------------------------------------------------
#En este caso yo utilicé la base de datos de Real Estate Price(Precio Inmoviliario)
#https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction
cdata = pd.read_csv('Real estate.csv')
#Visualizar tabla (Transpose data)
print('Datos Real estate Price T\n')
print(cdata.head().T)
#Visualizar su información
print('Información sobre el dataset\n')
print(cdata.info())

#Eliminar los valores que no son de importancia para la regresión lineal
cdata.drop(columns=['No','X1 transaction date'], inplace=True)
print('Base de datos sin las variables No y X1 transaction date\n')
print(cdata.head())

#Hace la función shuffle para desordenar los datos del dataset
shuffle_df = cdata.sample(frac=1).reset_index(drop=True)
X_original = shuffle_df.drop(columns=['Y house price of unit area'])
y = shuffle_df['Y house price of unit area']
#Visualizar las columnas de X y Y. 
#Donde X son todos los datos sobrantes excepto 'Y house price of unit area', ya que esa será nuestra label(y)
print('Valores de features x están mezcladas:\n')
print(X_original)
print('Valores del label y está mezclada:\n')
print(y)

#Normaliza los datos utilizando el escalador de datos--------------------------------------------------------
#El propósito de normalizar los datos es tratar de hacerlos más cercanos y evitar puntos de datos muy 
#   altos que puedan afectar las futuras predicciones del modelo entrenado.
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
dataScaler = StandardScaler()
scaler = dataScaler.fit(X_original)
X_scaled = scaler.transform(X_original)
# muestra el arreglo resultante
print(X_scaled)

# crea un dataframe con los datos normalizados
X = pd.DataFrame(X_scaled)
X.columns = X_original.columns
# muestra las primeras 5 líneas del dataframe resultante
print(X.head())

#Hacer el split de los datos de entrenamiento(train set) y evaluación(test set)
train_size = int(0.8 * len(cdata))

X_train = X[:train_size]
Y_train = y[:train_size]
X_test = X[train_size:]
Y_test = y[train_size:]

# Inicializar el peso y bias
w = np.zeros(X_train.shape[1])
b = 0.0

# Hyperparameters
alpha = 0.001  # Learning rate
epochs = 12000  # Number of epochs

# Train the model
w, b = train(X_train, Y_train, w, b, alpha, epochs)

# Evaluate the model on the test set
Y_pred = predict(X_test, w, b)

# Calculate and print the final loss on the test set
test_loss = avg_loss(X_test, Y_test, w, b)
print(f"Final Test Loss: {test_loss:.4f}")

#---------------------------------------------------------------------------------------------------

#Seleccionar el valor a utilizar en X para la regresión lineal 
X_train = X_train['X2 house age']
X_test = X_test['X2 house age']
#muestra los datos del feature de entrenamiento
print(X_train)

#Volver a iniciar los datos para los nuevos valores de X -------------------------------------------
# Inicializar el peso y bias
w = 0.0
b = 0.0
# Hyperparametros
alpha = 0.001  # Learning rate(alpha)
epochs = 12000  # Número de epochs(iteraciones)

# Entrenar al modelo
w, b = train(X_train, Y_train, w, b, alpha, epochs)
# Evaluar el modelo con el Test set
Y_pred = predict(X_test, w, b)

# Calculate and print the final loss on the test set
test_loss = avg_loss(X_test, Y_test, w, b)
print(f"Final Test Loss: {test_loss:.4f}")

# Visualize results for one feature (if you want to plot)
plt.scatter(X_test[:], Y_test, color='blue')  # Plotting first feature vs. target
plt.plot(X_test[:], predict(X_test, w, b), color='red')
plt.xlabel('First Feature')
plt.ylabel('Target')
plt.title('Linear Regression on Test Set')
plt.show()

def train_and_plot(X, y, w, b, alpha, epochs):
  '''Loops over multiple epochs and plot graphs showing progress'''
  for e in range(epochs):
    w, b = update_w_and_b(X, y, w, b, alpha)
  # plot visuals for last epoch
    if e == epochs-1:
      avg_loss_ = avg_loss(X, y, w, b)
      plt.scatter(X[:], y, color='blue')  # Plotting first feature vs. target
      plt.plot(X[:], np.dot(X, w) + b, color='red')
      plt.title("Epoch {} | Loss: {} | w:{}, b:{}".format(e, round(avg_loss_,2), round(w, 4), round(b, 4)))
      plt.show()
  return w, b

epoch_plots = [1, int(epochs/50), 2*int(epochs/50), 3*int(epochs/50), 4*int(epochs/50), epochs+1]
for epoch_plt in epoch_plots:
    w, b = train_and_plot(X_train, Y_train, 0.0, 0.0, alpha, epoch_plt)

#---------------------------------------------------------------------------------------------------
