import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageGrab, ImageOps
import numpy as np
import pickle
import cv2
import pandas as pd

# Activation functions
# ==========================
class Activation:
    def ReLU(X):
        return np.maximum(0, X)
    
    def softmaxMLP(X):
        # mathematically = e^X / sum(e^X)
        X_Shifted = X - np.max(X, axis=0, keepdims=True)
        e_X = np.exp(X_Shifted)
        A = e_X / np.sum(e_X, axis=0, keepdims=True)
        return A
    def softmax(X):
        # mathematically = e^X / sum(e^X)
        # shift if necassary
        #X = X - np.max(X)
        e_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        probs = e_X / np.sum(e_X, axis=1, keepdims=True)
        return probs

# Multilayer Perceptron
# ==========================
class MLP:
    def forward_prop(w1, b1, w2, b2, w3, b3, X):
    # Forward Propagation using ReLU activation function for the hidden layer and softmax for the output layer
        z1 = w1.dot(X) + b1
        a1 = Activation.ReLU(z1)
        z2 = w2.dot(a1) + b2
        a2 = Activation.ReLU(z2)
        z3 = w3.dot(a2) + b3
        a3 = Activation.softmaxMLP(z3)
        return z1, a1, z2, a2, z3, a3

# Convolutional Neural Network (Forward)
# ==========================
class CNN:
    class convLayer:
    # Convolutional Layer Forward Pass
    # ===============================
        def Forward(self, input):
            batch_size, in_channels, input_height, input_width = input.shape
            filter_height, filter_width = self.filters.shape[2:] # [2;] means take the last two elements of the shape
            self.num_filters = self.filters.shape[0]

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

                if n % 1000 == 0:
                    print(f"Convolution Sample {n}/{batch_size}")

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
    # ==========================
    class PoolingLayer:
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
                if n % 1000 == 0:
                    print(f"pooling Operation {n}/{batch_size}")
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
    # ==========================
    class FlattenLayer:
    # Flatten 
    # ===============================
        def Forward(self, input):
            # input of shape (batch_size, in_channels, height, width)
            self.cache = input.shape
            batch_size = input.shape[0]

            return input.reshape(batch_size, -1)
    # ==========================
    class denseLayer:
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

            return A
    # ==========================

    # Forward Pass for the CNN
    # ==========================
    def Init_Forward(cf1, cb1, cf2, cb2, fcW, fcb, outW, outb, stride, padding, channels, poolsize=2, poolstride=2):
        convLayer1 = CNN.convLayer()
        convLayer1.filters, convLayer1.bias, convLayer1.stride, convLayer1.padding, convLayer1.input_channels = cf1, cb1, stride, padding, channels
        poolLayer1 = CNN.PoolingLayer()
        poolLayer1.pool_size, poolLayer1.stride = poolsize, poolstride

        convLayer2 = CNN.convLayer()
        convLayer2.filters, convLayer2.bias, convLayer2.stride, convLayer2.padding, convLayer2.input_channels = cf2, cb2, stride, padding, channels
        poolLayer2 = CNN.PoolingLayer()
        poolLayer2.pool_size, poolLayer2.stride = poolsize, poolstride

        flatlayer = CNN.FlattenLayer()

        fcLayer = CNN.denseLayer()
        fcLayer.weights, fcLayer.bias = fcW, fcb

        outLayer = CNN.denseLayer()
        outLayer.weights, outLayer.bias = outW, outb

        layers = (convLayer1, poolLayer1, convLayer2, poolLayer2, flatlayer, fcLayer, outLayer)
        return layers

    def Forward(X, layers):
        convLayer1, poolLayer1, convLayer2, poolLayer2, flatlayer, fcLayer, outLayer = layers

        cL1_out = convLayer1.Forward(X)
        print(f"Shape of cL1_out: {cL1_out.shape}")
        # print(f"====================\n{cL1_out}\n====================")

        pL1_out = poolLayer1.Forward(cL1_out)
        print(f"Shape of pL1_out: {pL1_out.shape}")
        # print(f"====================\n{pL1_out}\n====================")

        cL2_out = convLayer2.Forward(pL1_out)
        print(f"Shape of cL2_out: {cL2_out.shape}")
        # print(f"====================\n{cL2_out}\n====================")

        pl2_out = poolLayer2.Forward(cL2_out)
        print(f"Shape of pl2_out: {pl2_out.shape}")
        # print(f"====================\n{pl2_out}\n====================")

        flat_out = flatlayer.Forward(pl2_out)
        print(f"Shape of flat_out: {flat_out.shape}")
        # print(f"====================\n{flat_out}\n====================")

        fc_out = fcLayer.Forward(flat_out)
        print(f"Shape of fc_out: {fc_out.shape}")
        # print(f"====================\n{fc_out}\n====================")

        a_out = outLayer.Forward(fc_out, activateSoftmax=True)
        print(f"output shape: {a_out.shape}")
        # print(f"====================\n{a_out}\n====================")
        return a_out
        #return a_out (softmax)
    # ==========================

class Measure:
    def Loss (a_out, Y):
        # Cross entropy loss function
        m = Y.shape[0]
        loss = -np.sum(Y * np.log(a_out + 1e-8)) / m # +1e-8 to avoid log(0)
        return loss

    def Accuracy (predictions, Y):
        # Calculate accuracy
        labels = np.argmax(Y, axis=1)
        accuracy = np.mean(predictions == labels) * 100
        return accuracy
    
def one_hot_encode(Y, num_classes=10):
    one_hot_y = np.zeros((Y.size, num_classes))
    one_hot_y[np.arange(Y.size), Y] = 1
    return one_hot_y

# Image Preprocessing
# ==========================
class ImgProc:
    def adaptive_threshold(pil_image, block_size=11, C=2):
        image_np = np.array(pil_image).astype('uint8')
        thresh = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, C)
        return Image.fromarray(thresh)
    
    def deskew(pil_image):
        image_np = np.array(pil_image)

        coords = np.column_stack(np.where(image_np > 0))

        if coords.shape[0] == 0:
            return pil_image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image_np.shape[:2]
        center = (w // 2, h // 2,)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)

    def denoise(pil_image, kernel_size=(3,3,)):
        image_np = np.array(pil_image)
        blurred = cv2.GaussianBlur(image_np, kernel_size, 0)    
        return Image.fromarray(blurred)

    def equalize_hist(pil_image):
        image_np = np.array(pil_image)
        equalized = cv2.equalizeHist(image_np)
        return Image.fromarray(equalized)
# ==========================

# Drawing App
# ==========================
class DrawingApp:
    def __init__(self, master, Modeltype, width=280, height=280, bg="white"):
        self.master = master
        self.width = width
        self.height = height
        self.bg = bg

        # Drawing Canvas (to handwrite numbers)
        self.canvas = Canvas(master, width=self.width, height=self.height, bg=self.bg)
        self.canvas.pack()

        # Predict Button
        self.button_predict = Button(master, text="Predict!", command=lambda: self.predict(Modeltype))
        self.button_predict.pack()
        self.button_clear = Button(master, text="Clear Canvas", command=self.clear_canvas)
        self.button_clear.pack()

        # label to display prediction results
        self.label_result = Label(master, text="Draw a digit and click \"Predict\"")
        self.label_result.pack()

        self.setup()

    def setup(self):
        # Variables to store mouse positions for drawing lines
        self.old_x = None
        self.old_y = None
        self.pen_width = 8 # Pen thickness

        # Bind mouse events for drawing
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def paint(self, event):
        paint_color = "black"
        if self.old_x is None or self.old_y is None:
            self.old_x = event.x
            self.old_y = event.y
        
        # Draw a line from the previous point to current point
        self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                     width=self.pen_width, fill=paint_color,
                                     capstyle=tk.ROUND, smooth=True, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y
    
    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.old_x, self.old_y = None, None
        self.label_result.config(text="Canvas cleared. Draw a digit and click \"Predict\"")

    def preprocess_drawing(self, Modeltype):
        self.canvas.update()
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        # Capture the Canvas content as an image
        img = ImageGrab.grab().crop((x, y, x1, y1))

        # convert to grayscale
        img = ImageOps.grayscale(img)
        img_eq = ImgProc.equalize_hist(img)
        img_denoised = ImgProc.denoise(img_eq)
        #img_deskewed = ImgProc.deskew(img_denoised)
        #img_thresh = ImgProc.adaptive_threshold(img_deskewed)

        # Invert Colors
        img_invert = ImageOps.invert(img_denoised)

        #Resize to 28x28
        img_resized = img_invert.resize((28, 28), Image.Resampling.LANCZOS)
        if (Modeltype == "CNN"):
            img_np = np.array(img_resized).astype("float32") / 255.0
            img_np = img_np.reshape(1, 1, 28, 28) # channels-first
            return img_np
        else:
            # Convert to numpy array and normalize to [0, 1]
            img_np = np.array(img_resized).astype("float32") / 255.0
            # Flatten the image
            img_np = img_np.flatten().reshape(784, 1)
            return img_np
    # return img_np

    def predict(self, Modeltype):
        # preprocess the drawing
        img_np = self.preprocess_drawing(Modeltype)

        if Modeltype == "MLP":
            # get the model parameters (assuemend to be set in self.model_params)
            w1, b1, w2, b2, w3, b3 = self.model_params
            # Update the result label with the prediction
            self.label_result.config(text=f"Prediction: {str(predict_MLP_sample(img_np, w1, b1, w2, b2, w3, b3))}")
        elif Modeltype == "CNN":
            channels = 0
            stride = 1
            padding = 1
            cf1, cb1, cf2, cb2, fcW, fcb, outW, outb = self.model_params
            self.label_result.config(text=f"Prediction: {str(predict_CNN_sample(img_np, cf1, cb1, cf2, cb2, fcW, fcb, outW, outb, stride, padding, channels)[0])}")

    def set_model_params(self, params):
        self.model_params = params
# ==========================

def load_model(filename):
    with open(filename, 'rb') as file:
        params = pickle.load(file)
    return params

def predict_MLP_sample(X, w1, b1, w2, b2, w3, b3): # X is image_flat
    _, _, _, _, _, a_out = MLP.forward_prop(w1, b1, w2, b2, w3, b3, X)
    prediction = np.argmax(a_out, axis=0)
    return prediction, a_out

def predict_CNN_sample(X, cf1, cb1, cf2, cb2, fcW, fcb, outW, outb, stride, padding, channels):
    layers = CNN.Init_Forward(cf1, cb1, cf2, cb2, fcW, fcb, outW, outb, stride, padding, channels)
    a_out = CNN.Forward(X, layers)
    #print(a_out)
    predictions = np.argmax(a_out, axis=1)
    print(f"Predictions:\n {predictions}\n=================")
    return predictions, a_out

def Initialize_test_Data(datapath, modeltype):
    data = pd.read_csv(datapath)

    data = np.array(data)
    np.random.shuffle(data)
    Y = data[:, 0] # Labes
    X = data[:, 1:] # pixels (num_samples, 784)
    # Normalize the pixel values

    X = X / 255.
    num_samples = X.shape[0]

    if modeltype == "CNN": 
        X = X.reshape(num_samples, 1, 28, 28) # channels-first

    return X, Y, num_samples

# THE ACTUAL MAIN
# ====================================
# Load the Model
# ========================
Modeltype = input("Modeltype? (1 for MLP, 2 for CNN):")
if Modeltype == '1':
    Modeltype = "MLP"
    Modelfile = input("Enter the Name of model to load (.pkl), otherwise Default will be loaded:")
    if Modelfile == "":
        Modelfile = r"C:\coding\Neural\py\NumberClassification\MLP\paramsets\NumClass_MBGD3.pkl"
        params = load_model(Modelfile)
    else:
        Modelfile = r"C:\coding\Neural\py\NumberClassification\MLP\paramsets\\" + Modelfile
        params = load_model(Modelfile)

    print(f"Model {Modelfile} loaded")

elif Modeltype == '2':
    Modeltype = "CNN"
    Modelfile = input("Enter the Name of model to load (.pkl), otherwise Default will be loaded:")
    if Modelfile == "":
        Modelfile = r"C:\coding\Neural\py\NumberClassification\CNN\paramsets\Conv_MBGD_2c2p2d_alp005.pkl"
        params = load_model(Modelfile)
    else:
        Modelfile = r"C:\coding\Neural\py\NumberClassification\CNN\paramsets\\" + Modelfile
        params = load_model(Modelfile)

    print(f"Model {Modelfile} loaded")

else:
    print("Invalid Modeltype")

Testtype = input("Run automated Test on MNIST_test or DIY? (auto/man):")
if Testtype == "auto":
    datapath = r"C:\coding\Neural\Datasets\mnist_test.csv\mnist_test.csv"
    X, Y, num_samples = Initialize_test_Data(datapath, Modeltype)
    X = X
    Y = Y
    one_hot_Y = one_hot_encode(Y, 10)

    if Modeltype == "CNN":
        cf1, cb1, cf2, cb2, fcW, fcb, outW, outb = params
        stride = 1
        padding = 1
        channels = 1
        predictions, a_out = predict_CNN_sample(X, cf1, cb1, cf2, cb2, fcW, fcb, outW, outb, stride, padding, channels)
        Loss = Measure.Loss(a_out, one_hot_Y)
        Accuracy = Measure.Accuracy(predictions, one_hot_Y)
        print(f"On the MNIST Test-Dataset, Architecture {Modeltype}, we achieved:\nLoss: {Loss:.8f}\nAccuracy: {Accuracy:.2f}%\n=================")

    elif Modeltype == "MLP":
        w1, b1, w2, b2, w3, b3 = params
        predictions, a_out = predict_MLP_sample(X.T, w1, b1, w2, b2, w3, b3)
        Loss = Measure.Loss(a_out.T, one_hot_Y)
        Accuracy = Measure.Accuracy(predictions, one_hot_Y)
        print(f"On the MNIST Test-Dataset, Architecture {Modeltype}, we achieved:\nLoss: {Loss:.8f}\n Accuracy: {Accuracy:.2f}%\n=================")

elif Testtype == "man":
    # Main 
    # ========================
    if __name__ == '__main__':
        root = tk.Tk()
        root.title("MNIST Digit Recognizer")

        # Drawing APP
        app = DrawingApp(root, Modeltype)

        # load the type of model, load the trained parameters, 
        app.set_model_params(params)

        # Run the Man Loop
        root.mainloop()
else:
    print("invalid Testype (auto/man)")