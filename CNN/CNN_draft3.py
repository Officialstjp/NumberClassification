import time as t
start_time = t.time()
import numpy as np
#import cupy as cp # if we want to rewrite in cupy later
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import cupy as cp, cupy.cuda.cudnn as cudnn
from cupy.cuda.memory import MemoryPool


# =========================================
# DOES NOT WORK
# =========================================


# Function definitions
# ===============================
def Initalize_Data(datapath, MakeDevData=True):
    data = pd.read_csv(datapath)

    data = cp.array(data)
    m, n = data.shape
    cp.random.shuffle(data)

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
    one_hot_y = cp.zeros((Y.size, num_classes))
    one_hot_y[cp.arange(Y.size), Y] = 1
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
    indices = cp.arange(m)
    cp.random.shuffle(indices)
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
        loss = -cp.sum(Y * cp.log(a_out + 1e-8)) / m # +1e-8 to avoid log(0)
        return loss

    def Accuracy (a_out, Y):
        # Calculate accuracy
        predictions = cp.argmax(a_out, axis=1)
        labels = cp.argmax(Y, axis=1)
        accuracy = cp.mean(predictions == labels) * 100
        return accuracy

class Activation:
    def ReLU(X):
        return cp.maximum(0, X)

    def softmax(X):
        # mathematically = e^X / sum(e^X)
        # shift if necassary
        #X = X - cp.max(X)
        e_X = cp.exp(X - cp.max(X, axis=1, keepdims=True))
        probs = e_X / cp.sum(e_X, axis=1, keepdims=True)
        return probs

    def ReLU_derivative(Z):
        return (Z > 0).astype(float)

class denseLayer:
# Initialize Parameters for dense layer
# ===============================
    def init_params(self, input_size, output_size):
        # Initialize weights and biases
        self.weights = cp.random.randn(output_size, input_size) * cp.sqrt(2. / input_size)
        self.bias = cp.zeros((output_size, 1))
        return self.weights, self.bias
    
# Dense Layer Forward Pass
# ===============================
    def Forward(self, input, activateSoftmax=False):
        # Forward pass for dense Layer, input is the output of the previous layer
        
        Z = cp.dot(input, self.weights.T) + self.bias.T
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
        dW = cp.dot(dZ.T, X) / m # dW shape = (output_size, input_size)
        db = cp.sum(dZ, axis=0, keepdims=True).T / m # db Shape = (output_size, 1)
        dInput = cp.dot(dZ, self.weights) # dX Shape = (15, 25)
        return dInput, dW, db
# return dInput, dW, db

class convLayer:
# Initialize Parameters for convolutional layer
# ===============================
    def init_params(self, num_filters, filter_size, input_channels, H, W, stride, padding):
        # Initialize Parameters for convolutional layer
        self.K = num_filters 
        self.fs = filter_size
        self.in_chan = input_channels
        self.stride = stride
        self.pad = padding
        # We use He initialization for the filter, because it prevents vanishing or exploding gradients (common for ReLU)
        self.filters = cp.random.randn(self.K, self.in_chan, self.fs, self.fs) * cp.sqrt(2. / (self.in_chan * self.fs * self.fs)) 
        self.bias = cp.zeros((self.K, 1)) # Bias for each filter

        self.handle = cudnn.get_handle() # Global
        self.wDesc = cudnn.create_filter_descriptor(self.filters)
        self.xDesc = cudnn.createTensorDescriptor(); cudnn.SetTensor4dDescriptor(self.xDesc, cudnn.CUDNN_TENSOR_NCHW, cp.float32, batch_size, self.in_chan, H, W)
        self.convDesc = cudnn.create_convolution_descriptor(pad=(self.pad, self.pad), 
                                                            stride=(self.stride, self.stride),
                                                            dilation=(1, 1), mode=cudnn.CUDNN_CROSS_CORRELATION, compute_type=cp.float32
        )

        self.ws_ptr = None
        self.ws_size = 0

        return self.filters, self.bias

# Convolutional Layer Forward Pass
# ===============================
    def Forward(self, x): #x = input
        N,C,H,W = x.shape
        _, _, FH, FW = self.filters.shape
        if (N,C,H,W) != getattr(self, "_cached_in_shape", None):
            # (Re) initialise # tensor descriptors
            self.xDesc = cudnn.createTensorDescriptor()
            cudnn.setTensor4dDescriptor(self.xDesc, cudnn.CUDNN_TENSOR_NCHW,
                                        cp.float32, N,C,H,W)
            
            OH = (H + 2*self.pad - FH) // self.stride + 1
            OW = (H + 2*self.pad - FW) // self.stride + 1
            self.yDesc = cudnn.createTesnordescriptor()
            cudnn.setTensor4dDescriptor(self.yDesc, cudnn.CUDNN_TENSOR_NCHW, cp.float32,
                                                     N, self.K, OH, OW)
            
            self._cached_in_shape = (N,C,H,W)

            # pick best algo once per shape
            self.fwd_algo = cudnn.get_convolution_forward_alogrithm(
                self.handle, self.xDesc, self.wDesc, self.convDesc,
                self.yDesc, cudnn.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0)
            
            # Query workspace size and grow if necassary
            size = cudnn.get_convolution_forward_workspace_size(
                self.handle, self.xDesc, self.wDesc, 
                self.convDesc, self.yDesc, self.fwd_algo)
            
            if size > self.ws_size:
                self.ws_ptr = cp.cuda.alloc(size)
                self.ws_size = size
            
        y = cp.empty((N, self.K, OH, OW), dtype=cp.float32)
        alpha, beta = 1.0, 0.0
        cudnn.convolution_forward(self.handle, alpha,
                                  self.xDesc, x.data.ptr,
                                  self.wDesc, self.W.data.ptr,
                                  self.convDesc, self.fwd_algo,
                                  self.ws_ptr.ptr, self.ws_size,
                                  beta,
                                  self.yDesc, y.data.ptr)
        
        # Bias and ReLU
        cudnn.addTensor(cp.cuda.get_current_stream().ptr,
                            alpha.data.ptr, self.biasDesc, self.bias.data.ptr,
                            alpha.data.ptr, self.yDesc, y.data.ptr)

        # ReLU
        if not hasattr(self, 'actDesc'):
            self.actDesc = cudnn.createActivationDescriptor()
            cudnn.setActivationDescriptor( self.actDesc, cudnn.CUDNN_ACTIVATION_RELU, cudnn.CUDNN_PROPAGATE_NAN, 0.0)
        cudnn.activationForward(cp.cuda.get_current_stream().ptr,
                                alpha.data.ptr, self.actDesc,
                                self.yDesc, y.data.ptr,
                                beta.data.ptr, self.yDesc, y.data.ptr)
        self.cache = (x, y)
        return y
# ========================================================
# ========================================================

# CONTINUE HERE

# ========================================================
# ========================================================
# return A

# Convolutional Layer Backward Pass
# ===============================
    def Backward(self, dA):
        """
        Compute gradients for the convolution layer.
        
        Args:
            dA (cp.ndarray): Gradient of the loss with respect to the output activation
                             (after the ReLU). It is expected as a NumPy array.
        
        Returns:
            dx (cp.ndarray): Gradient of the loss with respect to the layer input.
            
        Side-effects:
            Sets self.dw and self.db (gradients with respect to filters and bias)
            for subsequent update in backpropagation.
        """
        # Retrieve cached input and pre-activation output from forward pass.
        # 'x' is the original input, 'y' is the result from convolution+add bias (pre-ReLU).
        x, y = self.cache  # both expected to be CuPy arrays (GPU arrays)
        
        # Convert dA to a CuPy array (if it isn’t already) and ensure float32 type.
        dA = cp.asarray(dA, dtype=cp.float32)
        
        # Compute dZ: gradient of the convolution output before applying activation.
        # Here, we assume that the activation is ReLU.
        dZ = dA * (y > 0)  # elementwise multiplication
    
        handle = cudnn.get_handle()

        alpha = cp.float32(1.0)
        beta = cp.float32(0.0)
        # --- descriptors ---
        xDesc = cudnn.createTensorDescriptor(); cudnn.setTensor4dDescriptor(xDesc, cudnn.CUDNN_TENSOR_NCHW, cp.float32, x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        dZDesc = cudnn.createTensorDescriptor(); cudnn.setTensor4dDescriptor(dZDesc, cudnn.CUDNN_TENSOR_NCHW, cp.float32, dZ.shape[0], dZ.shape[1], dZ.shape[2], dZ.shape[3])
        wDesc = cudnn.create_filter_descriptor(self.filters, cp.float32)  # filter descriptor for the weights
        convDesc = cudnn.create_convolution_descriptor(pad=(self.pad, self.pad), stride=(self.stride, self.stride), dilation=(1, 1), mode=cudnn.CUDNN_CROSS_CORRELATION, compute_type=cp.float32)  # convolution descriptor

        # --- filter gradient ---
        dw = cp.empty_like(self.filters, dtype=cp.float32)
        algo_w = cudnn.get_convolution_backward_filter_algorithm(
            handle, xDesc, dZDesc, convDesc, wDesc,
            cudnn.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0)
        ws = cp.cuda.alloc(cudnn.get_convolution_backward_filter_workspace_size(
            handle, xDesc, dZDesc, convDesc, wDesc, algo_w))
        cudnn.convolution_backward_filter(handle, y,
                                          xDesc, x.data.ptr,
                                          dZDesc, dZ.data.ptr,
                                          convDesc, algo_w,
                                          ws.ptr, ws.size, beta,
                                          wDesc, dw.data.ptr)
        
        # --- data gradient ---
        dx = cp.empty_like(x, dtype=cp.float32)
        algo_x = cudnn.get_convolution_backward_data_algorithm(
            handle, wDesc, dZDesc, convDesc, xDesc,
            cudnn.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0)
        ws2 = cp.cuda.alloc(cudnn.get_convolution_backward_data_workspace_size(
            handle, wDesc, dZDesc, convDesc, xDesc, algo_x))
        cudnn.convolution_backward_data(handle, alpha,
                                        wDesc, cp.asarray(self.filters.data.ptr, dtype=cp.float32),
                                        dZDesc, dZ.data.ptr,
                                        convDesc, algo_x,
                                        ws2.ptr, ws2.size, beta,
                                        xDesc, dx.data.ptr)
        
        # --- bias gradient ---
        db = cp.zeros_like(self.bias, dtype=cp.float32)
        biasDesc = cudnn.createTensorDescriptor(); cudnn.setTensor4dDescriptor(biasDesc, cudnn.CUDNN_TENSOR_NCHW, cp.float32, 1, self.K, 1, 1)
        cudnn.add_tensor(handle, alpha, dZDesc, dZ.data.ptr, beta, biasDesc, db.reshape(1,-1,1,1).data.ptr)

        self.dw, self.db = dw, db
        return dx
# return dInput

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

        a_pool = cp.zeros((batch_size, channels, pool_h, pool_w))
        mask = cp.zeros_like(input, dtype=bool) # Mask to keep track of the max values

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
                        max_val = cp.max(patch)
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

        dInput = cp.zeros_like(input)
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

_global_pool = MemoryPool() 
cp.cuda.set_allocator(_global_pool.malloc) # Cupy will reuse memory

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

