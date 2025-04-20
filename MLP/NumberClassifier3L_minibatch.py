import time as t

start_time = t.time()

import pickle
import numpy as np
import os
#import cupy as cp # Offload to GPU, on PC
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r"C:\coding\Neural\Datasets\mnist_train.csv\mnist_train.csv")
#data = pd.read_csv(r"C:\Users\stefa\Onedrive\Dokumente\Coding\Python\NeuralNetworks\Datasets\mnist_train.csv\mnist_train.csv")

# Data processing, this converts the data into a numpy array
data = np.array(data)
# get the dataset dimensions (rows = m, columns = n)
m, n = data.shape
# shuffle the data
np.random.shuffle(data)

# Normalize the pixel values into the range [0, 1] and split the data into dev and train sets
data_dev = data[0:1000].T # .T = Transpose
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[0:m].T # .T = Transpose, 
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

class Layer:
    def init_params():
        # Initialize the weights and biases as random values
        hidden_size1 = 128
        hidden_size2 = 64
        w1 = np.random.randn(hidden_size1, 784) * np.sqrt(2. / 784)
        b1 = np.zeros((hidden_size1, 1))
        w2 = np.random.randn(hidden_size2, hidden_size1) * np.sqrt(2. / hidden_size1)
        b2 = np.zeros((hidden_size2, 1))
        w3 = np.random.randn(10, hidden_size2) * np.sqrt(2. / hidden_size2)
        b3 = np.zeros((10, 1))
        return w1, b1, w2, b2, w3, b3
    
    def forward_prop(w1, b1, w2, b2, w3, b3, X):
    # Forward Propagation using ReLU activation function for the hidden layer and softmax for the output layer
        z1 = w1.dot(X) + b1
        a1 = Activation.ReLU(z1)
        z2 = w2.dot(a1) + b2
        a2 = Activation.ReLU(z2)
        z3 = w3.dot(a2) + b3
        a3 = Activation.softmax(z3)
        return z1, a1, z2, a2, z3, a3
    
class Activation:
    def ReLU(X):
        return np.maximum(0, X)
    
    def softmax(X):
        # mathematically = e^X / sum(e^X)
        X_Shifted = X - np.max(X, axis=0, keepdims=True)
        e_X = np.exp(X_Shifted)
        A = e_X / np.sum(e_X, axis=0, keepdims=True)
        return A
    

class Measure:
    # Calculate accuracy
    def get_accuracy(predictions, Y):
        accuracy = np.sum(predictions == Y) / Y.size
        accuracy = accuracy * 100
        return accuracy

    # Get predictions
    def predict_proba(X, w1, b1, w2, b2, w3, b3):
        _, _, _, _, _, A3 = Layer.forward_prop(w1, b1, w2, b2, w3, b3, X)
        return A3

    def predictions(X, w1, b1, w2, b2, w3, b3):
        A3 = Measure.predict_proba(X, w1, b1, w2, b2, w3, b3)
        predictions = np.argmax(A3, 0)
        return predictions, A3

    # Loss function
    def calculate_loss(predictions, Y):
        predictions = predictions.T
        m = Y.shape[0]
        corect_log_probs = -np.log(predictions[np.arange(m), Y])
        loss = np.sum(corect_log_probs) / m
        return loss

class Optimization:
    def back_prop(z1, a1, z2, a2, z3, a3, w2, w3, Y, X):
        # This does the following:
        # 1. Calculate the gradients by comparing the predicted values with the actual values
        # 2. Update the weights and biases using the gradients
        
        # Calculate the gradients
        # Y = (10, m), the true values
        # a3 = (10, m), the predicted values, 
        # output layer gradients
        
        m = X.shape[1]
        # m = number of samples in mini-batch
        dZ3 = a3 - Y
        dW3 = 1/m * dZ3.dot(a2.T) #
        db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)

        # layer 2 gradients
        dZ2 = w3.T.dot(dZ3) * (z2 > 0)
        dW2 = 1/m * dZ2.dot(a1.T)
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

        # layer 1 gradients
        dZ1 = w2.T.dot(dZ2) * (z1 > 0)
        dW1 = 1/m * dZ1.dot(X.T)
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
        
        grads = [dW1, db1, dW2, db2, dW3, db3]
        clipped_grads = []
        for grad in grads:
            clipped_grads.append(clip_gradient(grad, 5))
        dW1, db1, dW2, db2, dW3, db3 = clipped_grads

        return dW1, db1, dW2, db2, dW3, db3
    def update_params(w1, b1, w2, b2, w3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
        # Update the weights and biases using the gradients
        w1 = w1 - alpha * dW1
        b1 = b1 - alpha * db1
        w2 = w2 - alpha * dW2
        b2 = b2 - alpha * db2
        w3 = w3 - alpha * dW3
        b3 = b3 - alpha * db3
        return w1, b1, w2, b2, w3, b3

def gradient_descent_training(X, Y, iterations, alpha):
    w1, b1, w2, b2, w3, b3, = Layer.init_params()
    one_hot_Y_train = one_hot_encode(Y_train)
    for i in range(iterations):
        z1, a1, z2, a2, z3, a3, = Layer.forward_prop(w1, b1, w2, b2, w3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = Optimization.back_prop(z1, a1, z2, a2, z3, a3, w2, w3, one_hot_Y_train, X)
        w1, b1, w2, b2, w3, b3 = Optimization.update_params(w1, b1, w2, b2, w3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        if i % 10 == 0:
            Predictions = Measure.predictions(X, w1, b1, w2, b2, w3, b3)
            CurrAccuracy = Measure.get_accuracy(Predictions, Y)

            A3 = Measure.predict_proba(X, w1, b1, w2, b2, w3, b3)
            CurrLoss = Measure.calculate_loss(A3, Y)

            print(f"Iterations: {i}, Accuracy: {CurrAccuracy:.2f}% Loss: {CurrLoss}")

        '''
        if i == 100:
            print("Switching learning rate to 0.05")
            alpha = 0.05
        '''
    return w1, b1, w2, b2, w3, b3

def get_minibatches(X, Y, batch_size):
    """
    Create minibatches of the data
    X: Input data of Shape (features, m)
    Y: Labels of shape (m,)
    Returns a list of tuples: (mini_X, mini_Y)
    """
    m = X.shape[1]
    indices = np.arange(m)
    np.random.shuffle(indices)
    mini_batches = []

    for i in range(0, m, batch_size):
        batch_indices = indices[i:i + batch_size]
        mini_X = X[:, batch_indices]
        mini_Y = Y[batch_indices]
        mini_batches.append((mini_X, mini_Y))

    return mini_batches

def mini_batch_gradient_descent(X, Y, epochs, alpha, batch_size):
    w1, b1, w2, b2, w3, b3 = Layer.init_params()

    for epoch in range(epochs):
        mini_batches = get_minibatches(X, Y, batch_size)
        for mini_X, mini_Y in mini_batches:
            one_hot_mini_Y = one_hot_encode(mini_Y)
            # Forward Propagation
            z1, a1, z2, a2, z3, a3 = Layer.forward_prop(w1, b1, w2, b2, w3, b3, mini_X)

            # Backpropagation
            # make sure shape of Y is (num_classes, m)
            
            dW1, db1, dW2, db2, dW3, db3 = Optimization.back_prop(z1, a1, z2, a2, z3, a3, w2, w3, one_hot_mini_Y, mini_X)

            # Update Parameters
            w1, b1, w2, b2, w3, b3 = Optimization.update_params(w1, b1, w2, b2, w3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        if epoch % 1 == 0:
            Predictions, A3 = Measure.predictions(X, w1, b1, w2, b2, w3, b3)
            epoch_accuracy = Measure.get_accuracy(Predictions, Y)
            epoch_loss = Measure.calculate_loss(A3, Y)
            print(f"Epoch: {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")
        if epoch % 50 == 0 and epoch != 0:
            print("Adjusting learning rate")
            alpha = alpha / 2

    return w1, b1, w2, b2, w3, b3

def one_hot_encode(Y, num_classes=10):
    one_hot = np.zeros((Y.size, num_classes))
    one_hot[np.arange(Y.size), Y] = 1
    return one_hot.T  # shape: (num_classes, m)

def clip_gradient(grad, clip_value):
    grad_norm = np.linalg.norm(grad)
    if grad_norm > clip_value:
        grad = grad * clip_value / grad_norm
    return grad

def save_model(filename, params):
    # Params is a tuple (w1, b1, w2, b2, w3, b3)
    with open(filename, 'wb') as file:
        pickle.dump(params, file)

def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = Measure.predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]

    print("Prediction: ", prediction)
    print("Label: ", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Train the model
# X = Training data, Y = Training labels, iterations = number of iterations, alpha = learning rate

# Best until now: 128 Neurons, Iterations = 100, Alpha = 0.10
# best until now: 1 hidden, 128 Neurons, Iterations = 200, Alpha = 0.10, = Acc 90.03, Loss: 0.3575
# 2 hidden, 128,128, iterations = 200, alpha = 0.1, Acc: 89.8948, Loss: 0.342765
# 2 hidden, 128, 64 iterations = 200, alpha = 0.1, After Training: Accuracy: 91.2531, Loss: 0.305758 (with proper Backprop)
# 2 hidden, 128, 64 I = 200, alpha = 0.2 After Training: Accuracy: 93.41989033150553, Loss: 0.2265549
# 2 hidden, 128, 128, I 200, alpha = 0.2 About the same as above
# With Mini-batch gradient descent, best:
#   2 hidden = 128, 64 Neurons, I = 150, alpha = 0.015 decreasing by /1.5 every 50 epochs,
#   batch size = 128, After Training: Accuracy: 99.74%, Loss: 0.01658

Iterations = 250
Alpha = 0.03
start_training_time = t.time()
print(f"Starting Training with {Iterations} iterations and a learning rate of {Alpha}")

#w1, b1, w2, b2, w3, b3 = gradient_descent_training(X_train, Y_train, Iterations, Alpha)

w1, b1, w2, b2, w3, b3 = mini_batch_gradient_descent(X_train, Y_train, Iterations, Alpha, 64)

Predictions, A3 = Measure.predictions(X_train, w1, b1, w2, b2, w3, b3)
CurrAccuracy = Measure.get_accuracy(Predictions, Y_train)
CurrLoss = Measure.calculate_loss(A3, Y_train)
print(f"After Training: Accuracy: {CurrAccuracy}, Loss: {CurrLoss}")

End_training_time = t.time()
print(f"Finished Training in {End_training_time - start_training_time:.2f}")

# Test the model
#for i in range(1):
    #test_prediction(i, w1, b1, w2, b2, w3, b3)

params = (w1, b1, w2, b2, w3, b3)
# wait for input to save the model

Saveinput = input("Save the model? (y/n):")
if Saveinput == 'y':
    filedir = r"C:\Users\stefa\Documents\Coding\Python\NeuralNetworks\NumberClassification\models"
    filename = input(f"The model-file will be saved to:\n{filedir}\n Please enter a filename:")
    if filename != "*.pkl":
        filename = f"{filename}.pkl"
    
    filedir = f"{filedir}\\{filename}"
        
    save_model(filedir, params)
    print("Model saved! at", filedir)