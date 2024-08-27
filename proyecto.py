import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Algoritmo para la regresión lineal
#---------------------------------------------------------------------------------------------------
# Function to update weights (w) and bias (b) during one epoch
def update_w_and_b(X, y, w, b, alpha):
    '''Update parameters w and b during 1 epoch'''
    dl_dw = 0.0
    dl_db = 0.0
    N = len(y)
    y_pred = np.dot(X, w) + b
    error = y - y_pred

    # Gradients for weight and bias
    dl_dw = -2 * np.dot(X.T, error) / N
    dl_db = -2 * np.sum(error) / N

    # Update weights and bias
    w -= alpha * dl_dw
    b -= alpha * dl_db

    return w, b

# Function to calculate the Mean Squared Error (MSE)
def avg_loss(X, y, w, b):
    '''Calculates the MSE'''
    N = len(y)
    y_pred = np.dot(X, w) + b
    total_error = np.sum((y - y_pred) ** 2)
    return total_error / N

# Training function
def train(X, y, w, b, alpha, epochs):
    '''Loops over multiple epochs and prints progress'''
    print('Training progress:')
    for e in range(epochs):
        w, b = update_w_and_b(X, y, w, b, alpha)
        if e % 400 == 0 or e == epochs - 1:
            avg_loss_ = avg_loss(X, y, w, b)
            print(f"Epoch {e} | Loss: {avg_loss_:.4f} | Weights: {w} | Bias: {b:.4f}")
    return w, b

# Prediction function
def predict(X, w, b):
    '''Make predictions based on the trained model'''
    return np.dot(X, w) + b

#---------------------------------------------------------------------------------------------------

#Cargar la base de datos
cdata = pd.read_csv('Real estate.csv')
#Visualizar tabla
print(cdata.head().T)
#Visualizar su información
print(cdata.info())

#Eliminar los valores que no son de importancia para la regresión lineal
cdata.drop(columns=['No','X1 transaction date'], inplace=True)
print(cdata.head())

#Hace la función shuffle para desordenar los datos del dataset
shuffle_df = cdata.sample(frac=1).reset_index(drop=True)
X_original = shuffle_df.drop(columns=['Y house price of unit area'])
y = shuffle_df['Y house price of unit area']
#Visualizar las columnas de X
print(X_original)
print(y)

# Normaliza los datos utilizando el escalador de datos
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

#Seleccionar el valor a utilizar en X para la regresión lineal
X_train = X_train['X2 house age']
X_test = X_test['X2 house age']

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
      x_list = np.array(range(0,50)) # Set x range
      y_list = (x_list * w) + b # Set function for the model based on w & b
      plt.scatter(x=X, y=y)
      plt.plot(y_list, c='r')
      plt.title("Epoch {} | Loss: {} | w:{}, b:{}".format(e, round(avg_loss_,2), round(w, 4), round(b, 4)))
      plt.show()
  return w, b

epoch_plots = [1, 2, 3, 11, 51, 101, epochs+1]
for epoch_plt in epoch_plots:
    w, b = train_and_plot(X_train, Y_train, 0.0, 0.0, alpha, epoch_plt)