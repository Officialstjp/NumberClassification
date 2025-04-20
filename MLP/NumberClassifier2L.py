import time as t

start_time = t.time()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r"C:\Users\stefa\Documents\Coding\Python\NeuralNetworks\Datasets\mnist_train.csv\mnist_train.csv")

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

# Initialize Parameters
class Layer:
    def init_params():
        hidden_size = 128
        w1 = np.random.randn(hidden_size, 784) * np.sqrt(2. / 784)
        b1 = np.zeros((hidden_size, 1))
        w2 = np.random.randn(10, hidden_size) * np.sqrt(2. / hidden_size)
        b2 = np.zeros((10, 1))
        return w1, b1, w2, b2
    
    def forward_prop(w1, b1, w2, b2, X):
        # X = (784, m)
        # w1 = (10, 784)
        # b1 = (10, 1)
        # w2 = (10, 10)
        # b2 = (10, 1)
        z1 = w1.dot(X) + b1
        a1 = Activation.ReLU(z1)
        z2 = w2.dot(a1) + b2
        a2 = Activation.softmax(z2)
        return z1, a1, z2, a2
    
class Activation:
    def ReLU(X):
        return np.maximum(0, X)
    
    def softmax(X):
        X_Shifted = X - np.max(X, axis=0, keepdims=True)
        e_X = np.exp(X_Shifted)
        A = e_X / np.sum(e_X, axis=0, keepdims=True)
        return A

class Optimization:
    def back_prop(z1, a1, z2, a2, w2, Y, X):
        # Back Propagation allows us to update the weights and biases according to the error
        # Y = (10, m), the true values

        # Calculate the gradients
        dZ2 = a2 - Y
        dW2 = 1/m * dZ2.dot(a1.T)
        db2 = 1/m * np.sum(dZ2)

        dZ1 = w2.T.dot(dZ2) * (z1 > 0)
        dW1 = 1/m * dZ1.dot(X.T)
        db1 = 1/m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha):
        # Update the weights and biases using the gradients
        w1 = w1 - alpha * dW1
        b1 = b1 - alpha * db1
        w2 = w2 - alpha * dW2
        b2 = b2 - alpha * db2
        return w1, b1, w2, b2

class Measure:
    # Calculate accuracy
    def get_accuracy(predictions, Y):
        accuracy = np.sum(predictions == Y) / Y.size
        accuracy = accuracy * 100
        return accuracy

    # Get predictions
    def predict_proba(X, w1, b1, w2, b2):
        _, _, _, A2 = Layer.forward_prop(w1, b1, w2, b2, X)
        return A2

    def predictions(X, w1, b1, w2, b2):
        A2 = Measure.predict_proba(X, w1, b1, w2, b2)
        predictions = np.argmax(A2, 0)
        return predictions

    # Loss function
    def calculate_loss(predictions, Y):
        predictions = predictions.T
        m = Y.shape[0]
        corect_log_probs = -np.log(predictions[np.arange(m), Y])
        loss = np.sum(corect_log_probs) / m
        return loss

def gradient_descent_training(X, Y, iterations, alpha):
    w1, b1, w2, b2 = Layer.init_params()
    one_hot_Y_train = one_hot_encode(Y_train)
    for i in range(iterations):
        z1, a1, z2, a2 = Layer.forward_prop(w1, b1, w2, b2, X)
        dW1, db1, dW2, db2 = Optimization.back_prop(z1, a1, z2, a2, w2, one_hot_Y_train, X)
        w1, b1, w2, b2 = Optimization.update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            Predictions = Measure.predictions(X, w1, b1, w2, b2)
            CurrAccuracy = Measure.get_accuracy(Predictions, Y)

            A2 = Measure.predict_proba(X, w1, b1, w2, b2)
            CurrLoss = Measure.calculate_loss(A2, Y)

            print(f"Iterations: {i}, Accuracy: {CurrAccuracy} Loss: {CurrLoss}")
    return w1, b1, w2, b2

# Testing and evaluation
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = Measure.predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]

    print("Prediction: ", prediction)
    print("Label: ", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def one_hot_encode(Y):
    one_hot = np.zeros((Y.size, Y.max() + 1))
    one_hot[np.arange(Y.size), Y] = 1
    return one_hot.T  # shape: (num_classes, m)


# Train the model
# X = Training data, Y = Training labels, iterations = number of iterations, alpha = learning rate
Iterations = 200
Alpha = 0.1 
# Best until now: 128 Neurons, Iterations = 100, Alpha = 0.10
# best until now: 128 Neurons, Iterations = 200, Alpha = 0.10, = Acc 90.03, Loss: 0.3575

print(f"Starting Training with {Iterations} iterations and a learning rate of {Alpha}")

w1, b1, w2, b2 = gradient_descent_training(X_train, Y_train, Iterations, Alpha)

Predictions = Measure.predictions(X_train, w1, b1, w2, b2)
CurrAccuracy = Measure.get_accuracy(Predictions, Y_train)

A2 = Measure.predict_proba(X_train, w1, b1, w2, b2)
CurrLoss = Measure.calculate_loss(A2, Y_train)

print(f"After Training: Accuracy: {CurrAccuracy}, Loss: {CurrLoss}")
End_training_time = t.time()
print(f"Finished Training in {End_training_time - start_time:.2f}")

# Test the model
for i in range(1):
    test_prediction(i, w1, b1, w2, b2)