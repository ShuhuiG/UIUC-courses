import numpy as np
import h5py
import time
import copy
from random import randint
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()
#Implementation of stochastic gradient descent algorithm
#number of inputs
num_inputs = 28*28
#number of outputs
num_outputs = 10
#number of units
num_units = 150
model = {}
model['W1'] = np.random.randn(num_units,num_inputs)/100
model['C'] = np.random.randn(num_outputs,num_units)/100
model['b1'] = np.random.randn(num_units)/100
model['b2'] = np.random.randn(num_outputs)/100
model_grads = copy.deepcopy(model)
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ
def forward(x,y, model):
    Z = np.dot(model['W1'], x) + model['b1']
    H = 1 / (np.exp(-Z) + 1)
    U = np.dot(model['C'], H) + model['b2']
    p = softmax_function(U)
    return {'Z': Z, 'H': H, 'U': U, 'p': p}
def backward(x,y,pa, model, model_grads):
    dU = 1.0 * pa['p']
    dU[y] = dU[y] - 1.0
    model_grads['b2'] = dU
    model_grads['C'] = np.dot(np.array([dU]).T, np.array([pa['H']]))
    delta = np.dot(model['C'].T, dU)
    Hd = np.multiply(pa['H'], (1 - pa['H']))
    deH = np.multiply(delta, Hd)
    model_grads['b1'] = deH
    model_grads['W1'] = np.dot(np.array([deH]).T, np.array([x]))
    return model_grads
import time
time1 = time.time()
LR = .01
num_epochs = 30
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        para = forward(x, y, model)
        prediction = np.argmax(para['p'])
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x,y,para, model, model_grads)
        model['C'] = model['C'] - LR * model_grads['C']
        model['b2'] = model['b2'] - LR * model_grads['b2']
        model['b1'] = model['b1'] - LR * model_grads['b1']
        model['W1'] = model['W1'] - LR * model_grads['W1']
    print(total_correct/np.float(len(x_train) ) )
time2 = time.time()
print(time2-time1)
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    para = forward(x, y, model)
    prediction = np.argmax(para['p'])
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )
