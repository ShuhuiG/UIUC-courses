import numpy as np
import h5py
import time
import copy
from random import randint
#from scipy import signal

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] ).reshape(60000,28,28)
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] ).reshape(10000,28,28)
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

#model initialization
#channel is 5
#filter size = 3
r = np.random.RandomState(123)
model = {}
model['K'] = r.randn(3, 3, 5)/x_train.shape[1]
model['W'] = r.randn(10,26,26,5)/x_train.shape[1]
model['b'] = np.zeros(10)
model_grads = copy.deepcopy(model)

#Implementation of stochastic gradient descent algorithm
def softmax(z):
    ZZ = np.exp(z)
    return ZZ/(sum(ZZ))

#relu
def relu(input):
    output = np.copy(input)
    output[input<0] = 0
    return output
def r_gradient(input):
    output = np.copy(input)
    output[input>=0] = 1
    output[input<0] = 0
    return output
#convolution layer
def Convlayer(ima_input, fil_input):
    convolution = np.zeros((ima_input.shape[0]-fil_input.shape[1]+1,
                            ima_input.shape[0]-fil_input.shape[1]+1,
                            fil_input.shape[2]))
    for i in range(0,fil_input.shape[2]):
        for x in range(0, ima_input.shape[0]-fil_input.shape[1]+1):
            for y in range(0, ima_input.shape[0]-fil_input.shape[1]+1):
                convolution[x, y, i] = np.sum(ima_input[x:x+fil_input.shape[1],y:y+fil_input.shape[1]]*fil_input[:,:,i])
    return convolution

#forward
def forward(x, y,model_pa):
    #convolution layer
    Z = Convlayer(x, model_pa['K'])
    H = relu(Z)
    U = np.tensordot(H,model_pa['W'],axes = ((0,1,2),(1,2,3))) + model_pa['b']
    p = softmax(U)
    return {'Z':Z, 'H':H, 'U':U, 'p':p}

#backward
def backward(x, y, forward_result, model, model_grads):
    dy = np.zeros(10)
    dy[y] = 1
    dU = -(dy - forward_result['p'])
    model_grads['b'] = dU
    model_grads['W'] = np.tensordot(dU, forward_result['H'], axes = 0)
    delta = np.tensordot(dU, model['W'], axes = 1)
    temp = np.multiply(r_gradient(forward_result['Z']),delta)
    model_grads['K'] = Convlayer(x,temp)
    return model_grads

time1 = time.time()
LR = .01
num_epochs = 3
for epochs in range(num_epochs):
    total_correct = 0
    for n in range( x_train.shape[0]):
        n_random = randint(0,x_train.shape[0]-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        para = forward(x,y,model)
        prediction = np.argmax(para['p'])
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x,y,para, model, model_grads)
        model['b'] = model['b'] - LR * model_grads['b']
        model['W'] = model['W'] - LR * model_grads['W']
        model['K'] = model['K'] - LR * model_grads['K']
    print(total_correct/np.float(len(x_train) ) )
time2 = time.time()
print(time2-time1)
######################################################
#test data
total_correct = 0
for n in range(x_test.shape[0]):
    y = y_test[n]
    x = x_test[n][:]
    para = forward(x, y, model)
    prediction = np.argmax(para['p'])
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(x_test.shape[0]) )
