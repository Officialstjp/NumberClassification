# NumberClassification in Python (/Cuda) - from scratch

This Repository contains different from-scratch implementations (no Pytorch / Tensorflow) for Computer-Vision (MLP, CNN) Neural Networks to perform number classification on the MNIST Dataset:
- **MLP/NumberClassifier2L.py**: Working Multi-Layer-Perceptron - CPU Processing (NumPy), Full-Batch Gradient Descent
  
- **MLP/NumberClassifier3L_minibatch.py**: Working Multi-Layer-Perceptron - CPU Processing (NumPy), Mini-Batch Gradient Descent, allows saving trained Parameters per .pkl
  
- **CNN/CNN_draft2.py**: Working Convolutional Neural Network - CPU Processing (NumPy), Mini-Batch Gradient Descent, allows saving trained Parameters per .pkl, Network architecture defined and adjustable internally

- **CNN/CNN_draft3.py**: Non-Functional Convolutional Neural Network Implementation - GPU Processing (CuPy, cuda.cudnn),
    -> From-scratch Handling of Descriptors, Memory Allocation, picks best conv-algorithm per tensor
    -> Not all functionality migrated to GPU-Processing, currently mixed state

- **ModelTestingNewData.py**: CLI-Utility for testing trained models on existing and new data

**No further work is planned in this Repository.**
