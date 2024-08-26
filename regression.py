def sigmoid_function(X):
  return 1/(1+math.e**(-X))

def log_regression4(X, y, alpha, epochs):
  y_ = np.reshape(y, (len(y), 1)) # shape (150,1)
  N = len(X)
  theta = np.random.randn(len(X[0]) + 1, 1) #* initialize theta
  X_vect = np.c_[np.ones((len(X), 1)), X] #* Add x0 (column of 1s)
  avg_loss_list = []
  loss_last_epoch = 9999999
  for epoch in range(epochs):
    sigmoid_x_theta = sigmoid_function(X_vect.dot(theta)) # shape: (150,5).(5,1) = (150,1)
    grad = (1/N) * X_vect.T.dot(sigmoid_x_theta - y_) # shapes: (5,150).(150,1) = (5, 1)
    best_params = theta
    theta = theta - (alpha * grad)
    hyp = sigmoid_function(X_vect.dot(theta)) # shape (150,5).(5,1) = (150,1)
    avg_loss = -np.sum(np.dot(y_.T, np.log(hyp) + np.dot((1-y_).T, np.log(1-hyp)))) / len(hyp)
    # if epoch % 50 == 0:
    #   print('epoch: {} | avg_loss: {}'.format(epoch, avg_loss))
    #   print('')
    avg_loss_list.append(avg_loss)
    loss_step = abs(loss_last_epoch - avg_loss) #*
    loss_last_epoch = avg_loss #*
    # if loss_step < 0.001: #*
    #   # print('\nStopping training on epoch {}/{}, as (last epoch loss - current epoch loss) is less than 0.001 [{}]'.format(epoch, epochs, loss_step)) #*
    #   break #*
    #plt.plot(np.arange(1, epoch+1), avg_loss_list[1:], color='red')
    #plt.title('Cost function')
    #plt.xlabel('Epochs')
    #plt.ylabel('Cost')
    #plt.show()
  return best_params