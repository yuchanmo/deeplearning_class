import numpy as np

def identity_function(x):
    return x

def relu(x):
    return np.maximum(0,x)

def step_function(x):
    return np.array(x>0,dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.argmax([0.1,0.2])
    return network

def forward(network,x,activatefunc=sigmoid):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1) + b1
    print(a1.shape)
    z1 = sigmoid(a1)
    print(z1.shape)
    a2 = np.dot(z1,W2) + b2
    print(a2.shape)
    z2 = sigmoid(a2)
    print(z2.shape)
    a3 = np.dot(z2,W3) + b3
    print(a3.shape)
    
    y = identity_function(a3)

    return y

net = init_network()
x = np.array([1.0,0.5])
y = forward(net,x)
print(y)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    
    sum_exp_a = np.sum(exp_a)
    y = exp_a/ sum_exp_a
    return y

a =np.array([0.3,2.9,4.0])
y = softmax(a)