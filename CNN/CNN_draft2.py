import time as t
start_time = t.time()
import numpy as np
#import cupy as cp # if we want to rewrite in cupy later
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# Function definitions
# ===============================
def Initalize_Data(datapath, MakeDevData=True):
    data = pd.read_csv(datapath)

    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    Y = data[:, 0] # Labes
    X = data[:, 1:] # pixels (num_samples, 784)
    # Normalize the pixel values
    X = X / 255.

    # Reshape
    num_samples = X.shape[0]
    X = X.reshape(num_samples, 1, 28, 28) # channels-first

    #  and split the data into dev and train sets
    if MakeDevData:
        # Split into dev and train sets, for example, using the first 1000 as dev:
        X_dev = X[:1000]
        Y_dev = Y[:1000]
        X_train = X[1000:]
        Y_train = Y[1000:]
        return X_train, Y_train, X_train.shape[0], X_dev, Y_dev
    else:
        return X, Y, num_samples

def one_hot_encode(Y, num_classes=10):
    one_hot_y = np.zeros((Y.size, num_classes))
    one_hot_y[np.arange(Y.size), Y] = 1
    return one_hot_y

def update_params(oldparameters, gradients, alpha):
    new_params = []
    for param, grad in zip(oldparameters, gradients):
        new_params.append(param - alpha * grad)
    return tuple(new_params)

def get_minibatches(X, Y, batch_size):
    """
    Create minibatches of the data
    X: Input data of Shape (num_samples, 1, 28, 28)
    Y: Labels of shape (m,)
    Returns a list of tuples: (mini_X, mini_Y)
    """
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    mini_batches = []

    for i in range(0, m, batch_size):
        batch_indices = indices[i:i + batch_size]
        mini_X = X[batch_indices]
        mini_Y = Y[batch_indices]
        mini_batches.append((mini_X, mini_Y))

    return mini_batches

def save_model(filename, params):
    # Params is a tuple (w1, b1, w2, b2, w3, b3)
    with open(filename, 'wb') as file:
        pickle.dump(params, file)

class Measure:
    def Loss (a_out, Y):
        # Cross entropy loss function
        m = Y.shape[0]
        loss = -np.sum(Y * np.log(a_out + 1e-8)) / m # +1e-8 to avoid log(0)
        return loss

    def Accuracy (a_out, Y):
        # Calculate accuracy
        predictions = np.argmax(a_out, axis=1)
        labels = np.argmax(Y, axis=1)
        accuracy = np.mean(predictions == labels) * 100
        return accuracy

class Activation:
    def ReLU(X):
        return np.maximum(0, X)

    def softmax(X):
        # mathematically = e^X / sum(e^X)
        # shift if necassary
        #X = X - np.max(X)
        e_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        probs = e_X / np.sum(e_X, axis=1, keepdims=True)
        return probs

    def ReLU_derivative(Z):
        return (Z > 0).astype(float)

class denseLayer:
# Initialize Parameters for dense layer
# ===============================
    def init_params(self, input_size, output_size):
        # Initialize weights and biases
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((output_size, 1))
        return self.weights, self.bias
    
# Dense Layer Forward Pass
# ===============================
    def Forward(self, input, activateSoftmax=False):
        # Forward pass for dense Layer, input is the output of the previous layer
        
        Z = np.dot(input, self.weights.T) + self.bias.T
        self.cache = input, Z
        if activateSoftmax:
            A = Activation.softmax(Z)
        else:
            A = Activation.ReLU(Z)

        return Z, A
# return Z, A

# Dense Layer Backward Pass
# For the output layer, dz needs to be the output of the network, and Y_true is the true labels
# ===============================
    def Backward(self, dz, is_output_layer=False, Y_true=None):
        X, Z = self.cache
        m = X.shape[0]
        if is_output_layer:
            dZ = (dz - Y_true) / m    
        else:
            dZ = dz * Activation.ReLU_derivative(Z)
        dW = np.dot(dZ.T, X) / m # dW shape = (output_size, input_size)
        db = np.sum(dZ, axis=0, keepdims=True).T / m # db Shape = (output_size, 1)
        dInput = np.dot(dZ, self.weights) # dX Shape = (15, 25)
        return dInput, dW, db
# return dInput, dW, db

class convLayer:
# Initialize Parameters for convolutional layer
# ===============================
    def init_params(self, num_filters, filter_size, input_channels, stride, padding):
        # Initialize Parameters for convolutional layer
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.stride = stride
        self.padding = padding
        # We use He initialization for the filter, because it prevents vanishing or exploding gradients (common for ReLU)
        self.filters = np.random.randn(num_filters, input_channels, filter_size, filter_size) * np.sqrt(2. / (input_channels * filter_size * filter_size)) 
        self.bias = np.zeros((num_filters, 1)) # Bias for each filter
        return self.filters, self.bias

# Convolutional Layer Forward Pass
# ===============================
    def Forward(self, input):
        batch_size, in_channels, input_height, input_width = input.shape
        filter_height, filter_width = self.filters.shape[2:] # [2;] means take the last two elements of the shape

        # Compute output dimensions
        output_height = (input_height + 2*self.padding - filter_height) // self.stride + 1
        output_width = (input_width + 2*self.padding - filter_width) // self.stride + 1
        Z = np.zeros((batch_size, self.num_filters, output_height, output_width))

        for n in range(batch_size):
            sample = input[n]
            if self.padding > 0:
                sample_padded = np.pad(sample, ((0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            else:
                sample_padded = sample

            # perform convolution
            for f in range(self.num_filters):
                #print(f"Performing Convolution for {f}-th filter")
                for i in range(output_height):
                    for j in range(output_width):
                        vert_start = i * self.stride
                        vert_end = vert_start + filter_height
                        horiz_start = j * self.stride
                        horiz_end = horiz_start + filter_width
                        # extract current patch
                        patch = sample_padded[:, vert_start:vert_end, horiz_start:horiz_end] # (channels, height, width)
                        Z[n, f, i, j] = np.sum(patch * self.filters[f] + self.bias[f])                  
        self.cache = (input, Z)
        A = Activation.ReLU(Z)
        return A
# return A

# Convolutional Layer Backward Pass
# ===============================
    def Backward(self, dz):
        X, Z = self.cache # X = (batch_size, channels, height, width)
        filter_height, filter_width = self.filters.shape[2:]
        batch_size, channels, output_height, output_width = dz.shape

        # Initialize gradients
        dZ = np.zeros_like(dz)
        dfilters = np.zeros_like(self.filters)
        dbias = np.zeros_like(self.bias)
        dInput = np.zeros_like(X)

        for n in range(batch_size):
            sample = X[n] # shape (channels, height, width)
            
            if self.padding > 0:
                sample_padded = np.pad(sample, ((0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            else:
                sample_padded = sample

            dInput_padded = np.zeros_like(sample_padded) # gradient with respect to the input

            # Compute gradients
            for f in range(self.num_filters):
                #print(f"Computing gradients for {f}-th filter")
                for i in range(output_height):
                    for j in range(output_width):
                        # determine the current patches indices
                        vert_start = i * self.stride
                        vert_end = vert_start + filter_height
                        horiz_start = j * self.stride
                        horiz_end = horiz_start + filter_width

                        # extract current patch
                        patch = sample_padded[:, vert_start:vert_end, horiz_start:horiz_end]
                        # compute gradient with respect to the filter
                        dZ[n, f, i, j] = dz[n, f, i, j] * Activation.ReLU_derivative(Z[n, f, i, j])
                        grad_value = dZ[n, f, i, j]

                        # Accumulate gradient with respect to the filter
                        
                        dfilters[f] += patch * grad_value

                        # Accumulate gradient with respect to the bias                
                        dbias[f] += grad_value            

                        # add the gradient of the filter, scaled by dZ, to the corresponding region of dInput
                        dInput_padded[:, vert_start:vert_end, horiz_start:horiz_end] += self.filters[f] * grad_value 

            # Remove padding from the gradient with respect to the input if needed
            if self.padding > 0:
                dInput[n] = dInput_padded[:, self.padding:-self.padding, self.padding:-self.padding]
            else:
                dInput[n] = dInput_padded

        return dInput, dfilters, dbias
# return dInput, dfilters, dbias

class PoolingLayer:
# Initialize Parameters for pooling layer
# ===============================
    def init_params(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        return self.pool_size, self.stride
    
# Pooling Layer Forward Pass
# ===============================
    def Forward(self, input):
        # Perform max pooling
        batch_size, channels, h, w = input.shape
        pool_h = (h - self.pool_size) // self.stride + 1 # '//' is integer division
        pool_w = (w - self.pool_size) // self.stride + 1

        a_pool = np.zeros((batch_size, channels, pool_h, pool_w))
        mask = np.zeros_like(input, dtype=bool) # Mask to keep track of the max values

        for n in range(batch_size):
            for c in range(channels):
                for i in range(pool_h):
                    for j in range(pool_w):
                        vert_start = i * self.stride
                        vert_end = vert_start + self.pool_size
                        horiz_start = j * self.stride
                        horiz_end = horiz_start + self.pool_size
                        # extract current patch
                        # print("For i =", i, "j =", j, "slice:", vert_start, "to", vert_end)
                        patch = input[n, c, vert_start:vert_end, horiz_start:horiz_end]
                        max_val = np.max(patch)
                        a_pool[n, c, i, j] = max_val

                        # Create mask for the max value
                        max_mask = (patch == max_val)
                        mask[n, c, vert_start:vert_end, horiz_start:horiz_end] = max_mask

        self.cache = (input, mask) # Cache the input and mask for backward pass
        return a_pool # (batch_size, channels, h, w)
# return a_pool

# Pooling Layer Backward Pass
# ===============================
    def Backward(self, dz):
        input, mask = self.cache
        #h, w = input.shape
        batch_size, channels, out_h, out_w = dz.shape
        # initialize gradient with respect to the input

        dInput = np.zeros_like(input)
        for n in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        vert_start = i * self.stride
                        vert_end = vert_start + self.pool_size
                        horiz_start = j * self.stride
                        horiz_end = horiz_start + self.pool_size

                        # get the gradient for the current pooled output
                        grad = dz[n, c, i, j]

                        mask_patch = mask[n, c, vert_start:vert_end, horiz_start:horiz_end]

                        # propagate the gradient to only the max value positions
                        dInput[n, c, vert_start:vert_end, horiz_start:horiz_end][mask_patch] = grad

        return dInput
# return dInput

class FlattenLayer:
# Flatten 
# ===============================
    def Forward(self, input):
        # input of shape (batch_size, in_channels, height, width)
        self.cache = input.shape
        batch_size = input.shape[0]
        
        return input.reshape(batch_size, -1)
# Unflatten
# ===============================
    def Backward(self, dz):
        # Reshape the gradient to the original shape of the input, self.cache is the original shape
        return dz.reshape(self.cache)

# Main 
# ===============================
''' Architecture
    Input (28,28) -> 28x28
    Conv Layer 1
    Pooling Layer 1 -> 14x14
    Conv Layer 2
    Pooling Layer 2 -> 7x7
    Flatten -> 49
    Fully Connected Layer (49 inputs, 49 outputs)
    Output (49 inputs, 10 outputs)
'''
# Data Parameters
# ===============================
datapath = r"C:\Coding\Neural\Datasets\mnist_train.csv\mnist_train.csv"     # Path to the dataset
MakeDevData = True    # If True, a dev dataset of a 1000 samples is created too

# How many input channels do we have?
# ================================
num_input_channels = 1 # 1 for grayscale, 3 for RGB

# Conv and Pooling Layer Parameters
# ===============================

# Convolutional Layer 1 Parameters
# ==============
num_filters1 = 12 # Number of filters in the first convolutional layer
filter_size1 = 3 # Size of the filters in the first convolutional layer
stride1 = 1 # Stride for the first convolutional layer
padding1 = 1 # Padding for the first convolutional layer

# Pooling Layer 1 Parameters
# ==============
pool_size1 = 2 # Size of the pooling window for the first pooling layer
pool_stride1 = 2 # Stride for the first pooling layer

# Convolutional Layer 2 Parameters
# ==============
num_filters2 = 24 # Number of fitlers in the second convolutional layer
filter_size2 = 3 # Size of the filters in the second convolutional layer
stride2 = 1 # Stride for the second convolutional layer
padding2 = 1 # Padding for the second convolutional layer

# Pooling Layer 2 Parameters
# ==============
pool_size2 = 2 # Size of the pooling window for the first pooling layer
pool_stride2 = 2 # Stride for the first pooling layer

# Dense Layer Parameters
# ==============
fc_input_size = 1176 # filters x height x width = 49 inputs
fc_output_size = 128 # 49 outputs
out_input_size = 128 # 49 inputs
out_output_size = 10 # 10 outputs

# Learning Rate
# ==============
learning_rate = 0.01

# Number of epochs
# ==============
epochs = 1  # Number of epochs for training

# ==============
batch_size = 128 # Number of samples per batch


# Initialize the data
X_Train, Y_train, m_train, X_dev, Y_dev = Initalize_Data(datapath, MakeDevData)
#X_Train, Y_train, m_train = Initalize_Data(datapath, MakeDevData=False)

# Initialize the classes for each layer
convLayer1 = convLayer()
conv_filters1, conv_bias1 = convLayer1.init_params(num_filters1, filter_size1, num_input_channels, stride1, padding1)

poolLayer1 = PoolingLayer()
pool_size1, pool_stride1 = poolLayer1.init_params(pool_size1, pool_stride1)

convLayer2 = convLayer()
conv_filters2, conv_bias2 = convLayer2.init_params(num_filters2, filter_size2, num_filters1, stride2, padding2)

poolLayer2 = PoolingLayer()
pool_size2, pool_stride2 = poolLayer2.init_params(pool_size2, pool_stride2)

flatLayer = FlattenLayer()

fcLayer = denseLayer()
fc_weights, fc_bias = fcLayer.init_params(fc_input_size, fc_output_size) # 49 inputs, 49 outputs

OutputLayer = denseLayer()
output_weights, output_bias = OutputLayer.init_params(out_input_size, out_output_size) # 49 inputs, 10 outputs


Y = Y_train[0:128]
X = X_Train[0:128]
#one_hot_y = one_hot_encode(Y_dev[0:100], num_classes=10)

# Training
# ======================================================
# ======================================================
# padding doesnt work yet, adjust code to work for batches

start_training_time = t.time()
# Initialized Parameters
# ===============================
print(f"Starting training with following parameters:\n================="
        f"\ndata has '{num_input_channels}' channels, shape of '{X_dev.shape}' and '{m_train}' samples"
        f"\nconvLayer1: '{num_filters1}' filters with size '{filter_size1}', '{stride1}' stride, '{padding1}' padding"
        f"\npoolLayer1: pool size '{pool_size1}, pool stride '{pool_stride2}"
        f"\nconvLayer2: '{num_filters2}' filters with size '{filter_size2}', '{stride2}' stride, '{padding2}' padding"
        f"\npoolLayer2: pool size '{pool_size1}', pool stride '{pool_stride2}'"
        f"\nfc Layer: input neurons: '{fc_input_size}', output neurons: '{fc_output_size}'"
        f"\noutput Layer: input neurons: '{out_input_size}', output neurons: '{out_output_size}'"
        f"\nlearning rate of '{learning_rate}'"
        f"\nbatch size: {batch_size}"
        f"\nepochs: '{epochs}'"
        )

for epoch in range(epochs): # 1 epoch for testing
    mini_batches = get_minibatches(X, Y, batch_size)
    Batches = len(mini_batches)
    Batchi = 0

    for mini_X, mini_Y in mini_batches:
        one_hot_mini_Y = one_hot_encode(mini_Y, num_classes=10)
        Batchi += 1

        # ===============================
        # Forward Pass
        # ===============================
        print(f"===========================\nForward Pass | Batch {Batchi}/{Batches} | Epoch {epoch+1}/{epochs}\n===========================")

        convLayer1_out = convLayer1.Forward(mini_X)
        print(f"Convolution 1 complete, output shape: {convLayer1_out.shape}")

        poolLayer1_out = poolLayer1.Forward(convLayer1_out)
        print(f"Pooling 1 complete, output shape: {poolLayer1_out.shape}")

        convLayer2_out = convLayer2.Forward(poolLayer1_out)
        print(f"Convolution 2 complete, output shape: {convLayer2_out.shape}")

        poolLayer2_out = poolLayer2.Forward(convLayer2_out)
        print(f"Pooling 2 complete, output shape: {poolLayer2_out.shape}")

        flattened_out = flatLayer.Forward(poolLayer2_out)
        print(f"flattening complete, output shape: {flattened_out.shape}")

        Fc_Z, fcLayer_out = fcLayer.Forward(flattened_out)
        print(f"fully connected complete, output shape: {fcLayer_out.shape}")

        out_z, predictions = OutputLayer.Forward(fcLayer_out, True)
        print(f"output complete, output shape: {predictions.shape}")

        print("Forward Pass Complete\n===========================")

        # ===============================
        # Loss and Accuracy
        # ===============================
        Loss = Measure.Loss(predictions, one_hot_mini_Y)
        Accuracy = Measure.Accuracy(predictions, one_hot_mini_Y)
        print(f"Batch {Batchi}/{Batches} | Epoch {epoch+1}/{epochs}, Loss: {Loss:.8f}, Accuracy: {Accuracy:.2f}%")

        # ===============================
        # Backward Pass
        # ===============================

        print(f"===========================\nBackpropagation | Batch {Batchi}/{Batches} | Epoch {epoch+1}/{epochs}\n===========================")
        output_dInput, output_dW, output_db = OutputLayer.Backward(predictions, is_output_layer=True, Y_true=one_hot_mini_Y)
        print(f"output complete, Shapes of dX: {output_dInput.shape}, dW: {output_dW.shape}, db: {output_db.shape}")

        fc_dInput, fc_dW, fc_db = fcLayer.Backward(output_dInput)
        print(f"FC complete, Shapes of dX: {fc_dInput.shape}, dW: {fc_dW.shape}, db: {fc_db.shape}")

        unflattened = flatLayer.Backward(fc_dInput)
        print(f"unflattening complete, shape: {unflattened.shape}")

        pool2_dInput = poolLayer2.Backward(unflattened)
        print(f"Pooling 2 complete, shape: {pool2_dInput.shape}")

        conv2_dInput, dfilters2, dbias2 = convLayer2.Backward(pool2_dInput)
        print(f"Convolution 2 complete, Shapes of dX: {conv2_dInput.shape}, dfilters: {dfilters2.shape}, db: {dbias2.shape}")
        
        pool1_dInput = poolLayer1.Backward(conv2_dInput)
        print(f"Pooling 1 complete, shape: {pool1_dInput.shape}")

        conv1_dInput, dfilters1, dbias1 = convLayer1.Backward(pool1_dInput)
        print(f"Convolution 1 complete, Shapes of dX: {conv1_dInput.shape}, dfilters: {dfilters1.shape}, db: {dbias1.shape}")
        
        print("Backward Pass Complete\n===========================")
        # Old parameters and corresponding gradients MUST line up
        oldparameters = convLayer1.filters, convLayer1.bias, convLayer2.filters, convLayer2.bias, fcLayer.weights, fcLayer.bias, OutputLayer.weights, OutputLayer.bias
        gradients = dfilters1, dbias1, dfilters2, dbias2, fc_dW, fc_db, output_dW, output_db

        ''' DEBUGGING
        print(f"=================\nShapes of the Parameters:"
            "\n================="
            f"\nconv_filters1: {conv_filters1.shape}, Gradient: {dfilters1.shape}"
            f"\nconv_bias1: {conv_bias1.shape}, Gradient: {dbias1.shape}"
            f"\nconv_filters2: {conv_filters2.shape}, Gradient: {dfilters2.shape}"
            f"\nconv_bias2: {conv_bias2.shape}, Gradient: {dbias2.shape}"
            f"\nfc_weights: {fc_weights.shape}, Gradient: {fc_dW.shape}"
            f"\nfc_bias: {fc_bias.shape}, Gradient: {fc_db.shape}"
            f"\noutput_weights: {output_weights.shape}, Gradient: {output_dW.shape}"
            f"\noutput_bias: {output_bias.shape}, Gradient: {output_db.shape}\n================="
            )
        '''
        conv_filters1, conv_bias1, conv_filters2, conv_bias2, fc_weights, fc_bias, output_weights, output_bias = update_params(oldparameters, gradients, learning_rate)
        convLayer1.filters = conv_filters1
        convLayer1.bias = conv_bias1
        convLayer2.filters = conv_filters2
        convLayer2.bias = conv_bias2
        fcLayer.weights = fc_weights
        fcLayer.bias = fc_bias
        OutputLayer.weights = output_weights
        OutputLayer.bias = output_bias
        print("---Parameters updated---")

print(f"After {Batchi}/{Batches} Batches | {epoch+1}/{epochs}  Epoch, Loss: {Loss:.8f}, Accuracy: {Accuracy:.2f}%")

end_time = t.time()
print(f"Finished in {end_time-start_time:.2f}s, Training took {end_time-start_training_time:.2f}s")

params = [conv_filters1, conv_bias1, conv_filters2, conv_bias2, fc_weights, fc_bias, output_weights, output_bias]

filedir = r"C:\coding\neural\py\NumberClassification\CNN\paramsets"
filename = "Conv_MBGD_2c2p2d_alp005_02.pkl"
if filename != "*.pkl":
    filename = f"{filename}.pkl"

filedir = f"{filedir}\\{filename}"
    
save_model(filedir, params)
print("Model automatically saved! at", filedir)

Saveinput = input("Save the parameters? (y/n):")
if Saveinput == 'y':
    filedir = r"C:\Coding\Neural\py\NumberClassification\models"
    filename = input(f"The model-file will be saved to:\n{filedir}\n Please enter a filename:")
    if filename != "*.pkl":
        filename = f"{filename}.pkl"
    
    filedir = f"{filedir}\\{filename}"
        
    save_model(filedir, params)
    print("Model saved! at", filedir)

