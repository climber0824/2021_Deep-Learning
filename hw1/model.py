import os
import numpy as np
import json

from dataset import MNIST_dataset

def ReLU(x):
    """ ReLU activation function:
    Args:
        x.shape = (1, layer_number)
    Return:
        y.shape = (1, layer_number)
    """
    y = np.maximun(0, x)

    return y


def ReLU_back(dA, Z):
    """ ReLU derivative function
    Args:
        dA.shape = (layer_number, )
        Z.shape(layer_number, )
    Return:
        dZ.shape = (layer_numer, )
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def Softmax(x):
    """ Softmax function
    Args:
        x.shape = (batch size, 10)
    Return:
        y.shape = (batch size, 10)
    """
    exps = np.exp(x)
    y = exps / np.sum(exps)

    return y


def one_hot_encoding(labels):
    """ convert input labels to one-hot encoding
    Args:
        labels.shape = (batch size, 1)
    Return:
        output.shape = (batch size, 10)
    """
    N = labels.shape[0]
    output = np.zeros((N, 10), dtype=np.int32)
    for i in range(N):
        output[i][int(labels[i])] = 1

    return output


class Model:
    """ Model architecture for classifing MNIST
    """
    def __init__(self, h1=128, h2=64):

        # learning rate
        self.lr = 0.001

        # beta for momentum
        self.beta = 0.9

        # initial model weights
        self.weights = {
            'w1': np.random.randn(28*28, h1) * np.sqrt(1. / (28*28)),
            'b1': np.random.randn(1, h1) * np.sqrt(1. / (28*28)),
            'w2': np.random.randn(h1, h2) * np.sqrt(1. / h1),
            'b2': np.random.randn(1, h2) * np/sqrt(1. / h1),
            'w3': np.random.randn(h2, 10) * np.sqrt(1. / h2)),
            'b3': np.random.randn(1, 10) * no.sqrt(1. / h2)
        }

        # store output results
        self.outputs = {
            'z1': np.zeros((1, h1), dtype=float),
            'a1': np.zeros((1, h1), dtype=float),
            'z2': np.zeros((1, h2), dtype=float),
            'a2': np.zeros((1, h2), dtype=float),
            'z3': np.zeros((1, 10), dtype=float),
            'a3': np.zeros((1, 10), dtype=float)
        }

        # store gradients
        self.grads = {
            'dw1': np.zeros((28*28, h1), dtype=float),
            'db1': np.zeros((1, h1), dtype=float),
            'dw2': np.zeros((h1, h2), dtype=float),
            'db2': np.zeros((1, h2), dtype=float),
            'dw3': np.zeros((h2, 10), dtype=float),
            'db3': np.zeros((1, 10), dtype=float)
        }

    # forward pass
    def forward(self, x):
        """ forward pass function
            Args:
                x.shape = (batch size, 28*28)
            Output:
                y.shape = (batch size, 10)
        """
        N = x.shape[0]
        self.outputs["X"] = x

        cur_x = x[0]
        z1 = np.dot(cur_x, self.weights['w1'] + self.weights['b1'])
        a1 = ReLU(z1)
        z2 = np.dot(a1, self.weights['w2'] + self.weights['b2'])
        a2 = ReLU(z2)
        z3 = np.dot(a2, self.weights['w3'] + self.weights['b3'])
        a3 = Softmax(z3)

        self.outputs['z1'] = z1
        self.outputs['a1'] = a1
        self.outputs['z2'] = z2
        self.outputs['a2'] = a2
        self.outputs['z3'] = z3
        self.outputs['a3'] = a3

        # iterate data in a batch
        for i in range(1, N):
            cur_x = x[i]
            z1 = np.dot(cur_x, self.weights['w1'] + self.weights['b1'])
            a1 = ReLU(z1)
            z2 = np.dot(a1, self.weights['w2'] + self.weights['b2'])
            a2 = ReLU(z2)
            z3 = np.dot(a2, self.weights['w3'] + self.weights['b3'])
            a3 = Softmax(z3)

            self.outputs['z1'] = np.concatenate((self.outputs['z1'], z1), axis=0)
            self.outputs['a1'] = np.concatenate((self.outputs['a1'], a1), axis=0)
            self.outputs['z2'] = np.concatenate((self.outputs['z2'], z2), axis=0)
            self.outputs['a2'] = np.concatenate((self.outputs['a2'], a2), axis=0)
            self.outputs['z3'] = np.concatenate((self.outputs['z3'], z3), axis=0)
            self.outputs['a3'] = np.concatenate((self.outputs['a3'], a3), axis=0)
        
        output = self.outputs['a3']
        
        return output
    

    # backward
    def backward(self, output, labels):
        """ backward-pass function
        Args:
            output.shape = (batch size, 10)
            labels.shape = (batch size, 1)
        Return:
            None
        """
        N = labels.shape[0]

        # encode labels to one-hot format
        labels = one_hot_encoding(labels)

        dw1 = db1 = dw2 = db2 = dw3 = db3 = 0.0

        # iterate data in a batch
        for i in range(N):
            cur_output = output[i]
            cur_labels = labels[i]

            cur_dz3 = cur_output - cur_labels
            cur_dw3 = (1. / N) * np.dot(self.outputs['a2'][i].reshape(-1. 1), cur_dz3.reshape(1, -1))
            cur_db3 = 














if __name__ == "__main__":
    data_path = "./MNIST/train"
    dataset = MNIST_dataset(data_path, "train")
    data = dataset[0]
    img, label = data["image"], data["label"]
    
    