# -*- coding: utf-8 -*-

"""
Este código esta basado en el curso de DeepLearning del profesor Andrew Ng
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
import tables
import numpy as np
from random import shuffle
from math import ceil
import cv2

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
    

def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b2", ..., "WL-1", "bL":
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    b(l+1) -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)            # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### ( 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l-1],layers_dims[l]) * 0.01
        parameters['b' + str(l+1)] = np.zeros((layers_dims[l], 1))

        ### END CODE HERE ###
        
        assert(parameters['W' + str(l)].shape == (layers_dims[l-1],layers_dims[l]))
        assert(parameters['b' + str(l+1)].shape == (layers_dims[l], 1))

        
    return parameters
    
    
def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)# integer representing the number of layers
     
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l-1],layers_dims[l]) * (2/layers_dims[l])**(0.5)
        parameters['b' + str(l+1)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###
        
    return parameters

    
def load_dataset_mnist():
    
    train_dataset = h5py.File('mnist.h5', "r")
    train_set_x_orig = np.array(train_dataset["train"]["inputs"]) # your train set features
    train_set_y_orig = np.array(train_dataset["train"]["targets"]) # your train set labels
    
    test_set_x_orig = np.array(train_dataset["test"]["inputs"]) # your test set features
    test_set_y_orig = np.array(train_dataset["test"]["targets"]) # your test set labels
    
    train_y=np.zeros((10,len(train_set_y_orig)))
    test_y=np.zeros((10,len(test_set_y_orig)))
    
    for i in range(0,10):
        train_y[i,np.where(train_set_y_orig[:]==i)[0]]=1
        test_y[i,np.where(test_set_y_orig[:]==i)[0]]=1
       
    idx_train = np.arange(train_y.shape[1])
    np.random.shuffle(idx_train)
    
    idx_test = np.arange(test_y.shape[1])
    np.random.shuffle(idx_test)
    
    train_set_x_orig = train_set_x_orig.reshape((train_set_x_orig.shape[0], train_set_x_orig.shape[1], train_set_x_orig.shape[2]))    
    test_set_x_orig = test_set_x_orig.reshape((test_set_x_orig.shape[0], test_set_x_orig.shape[1], test_set_x_orig.shape[2]))

    train_dataset.close()
    
    return train_set_x_orig[idx_train, ...], train_y[:,idx_train], test_set_x_orig[idx_test, ...], test_y[:,idx_test]


def load_dataset_mnist_2classes():
    
    train_dataset = h5py.File('mnist.h5', "r")
    train_set_x_orig = np.array(train_dataset["train"]["inputs"]) # your train set features
    train_set_y_orig = np.array(train_dataset["train"]["targets"]) # your train set labels
    
    test_set_x_orig = np.array(train_dataset["test"]["inputs"]) # your test set features
    test_set_y_orig = np.array(train_dataset["test"]["targets"]) # your test set labels
    
    
    ##### Train ###############################
    idx_0=np.where(train_set_y_orig[:]==0)[0]
    train_y_idx_0= train_set_y_orig[idx_0]
    train_x_idx_0= train_set_x_orig[idx_0,:,:]
    
    idx_1=np.where(train_set_y_orig[:]==1)[0] 
    train_y_idx_1= train_set_y_orig[idx_1]
    train_x_idx_1= train_set_x_orig[idx_1,:]
    
    train_set_x_orig=train_x_idx_0
    train_set_y_orig=train_y_idx_0
    
    train_set_x_orig=np.concatenate((train_set_x_orig, train_x_idx_1), axis=0)
    train_set_y_orig=np.concatenate((train_set_y_orig, train_y_idx_1), axis=0)
    train_set_y_orig=train_set_y_orig[:,np.newaxis]
    
    train_set_x_orig=train_set_x_orig.reshape(train_set_x_orig.shape[0],train_set_x_orig.shape[1]*train_set_x_orig.shape[2])
    train_set_x_orig=np.concatenate((train_set_x_orig, train_set_y_orig), axis=1)
    
    np.random.shuffle(train_set_x_orig)
    
    train_set_y_orig=train_set_x_orig[:,train_set_x_orig.shape[1]-1]
    train_set_x_orig=train_set_x_orig[:,0:train_set_x_orig.shape[1]-1]
        
    train_set_x_orig = train_set_x_orig.reshape((train_set_x_orig.shape[0], 28, 28))
    
    
    ############ Test ####################################
    
    idx_0=np.where(test_set_y_orig[:]==0)[0]
    test_y_idx_0= test_set_y_orig[idx_0]
    test_x_idx_0= test_set_x_orig[idx_0,:,:]
    
    idx_1=np.where(test_set_y_orig[:]==1)[0] 
    test_y_idx_1= test_set_y_orig[idx_1]
    test_x_idx_1= test_set_x_orig[idx_1,:]
    
    test_set_x_orig=test_x_idx_0
    test_set_y_orig=test_y_idx_0
    
    test_set_x_orig=np.concatenate((test_set_x_orig, test_x_idx_1), axis=0)
    test_set_y_orig=np.concatenate((test_set_y_orig, test_y_idx_1), axis=0)
    test_set_y_orig=test_set_y_orig[:,np.newaxis]
    
    test_set_x_orig=test_set_x_orig.reshape(test_set_x_orig.shape[0],test_set_x_orig.shape[1]*test_set_x_orig.shape[2])
    test_set_x_orig=np.concatenate((test_set_x_orig, test_set_y_orig), axis=1)
    
    np.random.shuffle(test_set_x_orig)
    
    
    test_set_y_orig=test_set_x_orig[:,test_set_x_orig.shape[1]-1]
    test_set_x_orig=test_set_x_orig[:,0:test_set_x_orig.shape[1]-1]
    
    test_set_x_orig = test_set_x_orig.reshape((test_set_x_orig.shape[0], 28, 28))
    
    #################################################

#    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
#    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    train_dataset.close()
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig 

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
#    hdf5_file.close()
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    
    


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y))/m))
        
    return p

def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3>0.5)
    return predictions

def load_dataset():
    
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
#    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
#    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    
    return train_X, train_Y, test_X, test_Y
    
    

def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
    
def load_data_aplication():
    
    hdf5_path = 'datasetGestos.hdf5'
    subtract_mean = False
    # open the hdf5 file
    hdf5_file = h5py.File(hdf5_path, "r")
    # subtract the training mean
    if subtract_mean:
        mm = hdf5_file["train_mean"][0, ...]
        mm = mm[np.newaxis, ...]
    # Total number of samples
    data_num = hdf5_file["train_img"].shape[0]
    
    train_img = np.uint8(hdf5_file["train_img"][0:data_num, ...])
    val_img = np.uint8(hdf5_file["val_img"][0:data_num, ...])
    test_img = np.uint8(hdf5_file["test_img"][0:data_num, ...])
    
    train_labels = np.uint8(hdf5_file["train_labels"][0:data_num, ...])
    val_labels = np.uint8(hdf5_file["val_labels"][0:data_num, ...])
    test_labels = np.uint8(hdf5_file["test_labels"][0:data_num, ...])
    
    hdf5_file.close()
       
    return train_img, train_labels[np.newaxis,:], test_img, test_labels[np.newaxis,:], np.array(['A', 'D'])


#def inicializar_Parametros(n_x, n_h, n_y):
      
#def inicializar_parametros_profundos(layer_dims):
   
#def linear_forward(A, W, b):
  
#def linear_activation_forward(A_prev, W, b, activation):
    
#def L_model_forward(X, parameters):
   
#def compute_cost(AL, YS, costFuntion):
   
#def linear_backward(dZ, cache):
   
#def linear_activation_backward(dA, cache, activation):
   
#def L_model_backward(AL, YS, caches, costFuntion):

#def update_parameters(parameters, grads, learning_rate):
  