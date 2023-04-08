"""
make this into a jupyter notebook thing, learn how to use that
"""

import idx2numpy
from utils import *

#################################################################################################################################################################################
#########################################################################       GET DATA         ################################################################################
#################################################################################################################################################################################


train_images = idx2numpy.convert_from_file("data/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("data/train-labels.idx1-ubyte")
test_images = idx2numpy.convert_from_file("data/test-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("data/test-labels.idx1-ubyte")

train_images = np.reshape(train_images, (train_images.shape[0], -1)) / 255
train_labels = np.reshape(train_labels, (train_labels.shape[0], -1))
test_images = np.reshape(test_images, (test_images.shape[0], -1)) / 255
test_labels = np.reshape(test_labels, (test_labels.shape[0], -1))

"""
4 layers:
0. input    (784)
1. h1       (16)
2. h2       (16)
3. output   (10)
"""

#################################################################################################################################################################################
#########################################################################       TRAIN NN         ################################################################################
#################################################################################################################################################################################

def train_example(input_layer, activations, w, ideal_output):
    # layer -1 is input layer
    
    activations[0] = input_layer
    activations[0] = activations[0].reshape((784,1))

    z = [
        np.zeros((16,1)),
        np.zeros((16,1)),
        np.zeros((10,1)),
    ]

    for l in [0, 1, 2]: 
        a_l = activations[l]    # 784 x 1
        w_l = w[0][l]           # 16 x 784
        b_l = w[1][l]           # 16 x 1

        z[l] = np.matmul(w_l, a_l) + b_l

        activations[l+1] = activation_function(z[l])

    # finding last layer's delta
    del_C_wrt_a = ideal_output - activations[3]
    delta = [
        np.zeros((16, 1)),
        np.zeros((16, 1)),
        np.zeros((10, 1))
    ]

    delta[2] = del_C_wrt_a * activation_function_prime(z[2]) # last layer

    # find delta for each layer
    for l in [1, 0]:
        w_next_l = w[0][l+1]
        delta[l] = np.matmul(w_next_l.T, delta[l+1]) * activation_function_prime(z[l])
    
    return delta  # 3 rows for each layer

def train_dataset(train_images, train_labels, activations, w, lr):
    """not the most efficient way"""
    deltas = [] # has delta for each training sample
    m = 0 # size of "deltas" array

    # getting delta for each training sample
    for i in range(len(train_images)):
        deltas.append(train_example(train_images[i], activations, w, train_labels[i]))
        m += 1

    # gradient decent
    """
    1. loop through l = L, L-1, ..., 2
    L = 4
    in py => 3, 2, 1

    """
    for l in [3, 2, 1]:
        sum_d = np.zeros(w[0][l-1].shape)   # here l-1 doesnt mean (l-1)th layer, its the index where the weights are stored
        sum_d_a = np.zeros((w[0][l-1].shape)) 

        for i in range(m):  # for each training example
            sum_d = deltas[i][l-1]  # our lth layer is computer's (l-1)th index

            """
            dim(deltas[i][l-1].T) = 1x784   for l = 1
            dim(deltas[i][l-1]) =   16x1    for l = 1
            """
            # sum_d_a = np.matmul(activations[l-1].T, deltas[i][l-1])

            sum_d_a = np.matmul(deltas[i][l-1], activations[l-1].T)

        w[0][l-1] = w[0][l-1] - (lr / m) * sum_d_a
        w[1][l-1] = w[1][l-1] - (lr / m) * sum_d


#################################################################################################################################################################################
#########################################################################        TEST NN         ################################################################################
#################################################################################################################################################################################

def test_nn(test_inputs, test_labels, w, activations):
    count = 0

    for i in range(len(test_inputs)):
        activations[0] = test_inputs[i]
        activations[0] = activations[0].reshape((784,1))

        z = [
            np.zeros((16,1)),
            np.zeros((16,1)),
            np.zeros((10,1)),
        ]

        for l in [0, 1, 2]: 
            a_l = activations[l]    # 784 x 1
            w_l = w[0][l]           # 16 x 784
            b_l = w[1][l]           # 16 x 1

            z[l] = np.matmul(w_l, a_l) + b_l

            activations[l+1] = activation_function(z[l])

        # print(activations[3])
        # print(np.argmax(activations[3]), test_labels[i])

        if np.argmax(activations[3]) == test_labels[i]:
            count += 1

    print(count / len(test_inputs), len(test_inputs))


#################################################################################################################################################################################
#####################################################################       FINAL OUTPUT         ################################################################################
#################################################################################################################################################################################


if __name__ == "__main__":
    """
    w_pq is a matrix with dim = len(p) x len(q)

    not really this: need to rewrite
    w_pq[i, j] = weight of (a_i from layer p and a_j of layer q)
    """

    # apart from input layer
    activations = [
        np.zeros((784,1)),# input layer
        np.zeros((16,1)), # hidden layer 1
        np.zeros((16,1)), # hidden layer 2
        np.zeros((10,1))  # output layer
    ]

    w_01 = np.random.uniform(low=-1, high=1, size=(16,784))
    w_12 = np.random.uniform(low=-1, high=1, size=(16,16))
    w_23 = np.random.uniform(low=-1, high=1, size=(10,16))

    b_1 = np.random.uniform(low=-1, high=1, size=(16,1))
    b_2 = np.random.uniform(low=-1, high=1, size=(16,1))
    b_3 = np.random.uniform(low=-1, high=1, size=(10,1))

    w = [
        [w_01, w_12, w_23], 
        [b_1, b_2, b_3]
    ]

    train_dataset(train_images, train_labels, activations, w, lr=0.01)
    test_nn(test_images, test_labels, w, activations)