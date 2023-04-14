"""
make this into a jupyter notebook thing, learn how to use that
"""

import idx2numpy
import numpy as np

#################################################################################################################################################################################
#########################################################################         UTILS        ##################################################################################
#################################################################################################################################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

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

# initialises variables: "activations" and "w"
def init_vars():
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

    return activations, w

#################################################################################################################################################################################
#########################################################################       TRAIN NN         ################################################################################
#################################################################################################################################################################################

def train_example(input_layer, activations, w, ideal_output):
    """
    For each training example, we must go through each of the following steps:
    1. Feedforward: 
        - find z and a for each layer (layer indices: 1, 2, 3)
    2. Output error:
        - delta for last layer (the output layer)
    3. Backpropogate the error:
        - find delta for each layer with layer indices: 2, 1
    """
    
    # activations array has 3 elements, each is a layer's activation values
    activations[0] = input_layer  
    activations[0] = activations[0].reshape((784,1))

    # initialise z
    z = [
        np.zeros((16,1)),
        np.zeros((16,1)),
        np.zeros((10,1)),
    ]

    ################################################ FEEDFORWARD #######################################################
    for l in [0, 1, 2]: # l here is (l_index - 1) since thats how things are stored in other arrays
        a_l = activations[l]
        w_l = w[0][l]
        b_l = w[1][l]

        z[l] = np.matmul(w_l, a_l) + b_l

        activations[l+1] = sigmoid(z[l])  
        # here we actually did find lth layer's z, a 

    ################################################ OUTPUT ERROR #######################################################
    del_C_wrt_a = ideal_output.reshape((10,1)) - activations[3]  # from defn

    # initialise deltas
    deltas = [
        np.zeros((16, 1)),
        np.zeros((16, 1)),
        np.zeros((10, 1))
    ]

    deltas[2] = del_C_wrt_a * sigmoid_prime(z[2]) # last layer

    ################################################ BACKPROPOGATION #####################################################
    for l in [1, 0]:  # here l is (l_index - 1)
        w_next_l = w[0][l+1]
        deltas[l] = np.matmul(w_next_l.T, deltas[l+1]) * sigmoid_prime(z[l])
    
    return deltas  # 3 rows for each layer

def train_dataset(train_images, train_labels, activations, w, lr):
    m = len(train_images)
    deltas = [] # has deltas array for each training sample (yeah deltas in deltas)

    # getting deltas for each training sample
    for i in range(m):
        deltas.append(train_example(train_images[i], activations, w, train_labels[i]))

    # gradient descent
    for l in [3, 2, 1]:
        sum_d_a = np.zeros((w[0][l-1].shape)) 
        sum_d = np.zeros(w[1][l-1].shape)

        ####################### FINDING SUMS #####################################
        for i in range(m):  # for each training example
            sum_d_a += np.matmul(deltas[i][l-1], activations[l-1].T)
            sum_d   += deltas[i][l-1]  # our lth layer is computer's (l-1)th index

        w[0][l-1] -= (lr / m) * sum_d_a 
        w[1][l-1] -= (lr / m) * sum_d


#################################################################################################################################################################################
#########################################################################        TEST NN         ################################################################################
#################################################################################################################################################################################

def test_nn(test_inputs, test_labels, w, activations):
    count = 0
    total = len(test_inputs)

    for i in range(len(test_inputs)):
        activations[0] = test_inputs[i]
        activations[0] = activations[0].reshape((784,1))

        z = [
            np.zeros((16,1)),
            np.zeros((16,1)),
            np.zeros((10,1)),
        ]

        for l in [0, 1, 2]: 
            a_l = activations[l]
            w_l = w[0][l]
            b_l = w[1][l]
            
            z[l] = np.matmul(w_l, a_l) + b_l

            activations[l+1] = sigmoid(z[l])

        if np.argmax(activations[3]) == np.argmax(test_labels[i]):
            count += 1

    print(f"accuracy = {count*100 / total}% i.e. {count} out of {total}")


#################################################################################################################################################################################
#####################################################################       FINAL OUTPUT         ################################################################################
#################################################################################################################################################################################

def one_hot_encode(labels):
    result = np.zeros((labels.shape[0], 10))
    for i, label in enumerate(labels):
        result[i, label] = 1
    return result

if __name__ == "__main__":
    activations, w = init_vars()

    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)

    test_nn(test_images, test_labels, w, activations)
    train_dataset(train_images, train_labels, activations, w, lr=0.01)
    
    for _ in range(1):
        # alternate approach: 
        # give subsets of training data set in each iteration 

        # generate a permutation of indices
        perm = np.random.permutation(len(train_images))

        # use the permutation to shuffle both arrays
        train_images = train_images[perm]
        train_labels = train_labels[perm]

        train_dataset(train_images, train_labels, activations, w, lr=0.01)    
        test_nn(test_images, test_labels, w, activations)