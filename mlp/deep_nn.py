import numpy as np
import time
import sys
import math

from models import Sequential
from layers import Dense


def seconds_to_string(s):
    """Returns a nicely formatted string in the form 00d 00h 00m 00s"""
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    string = '%ds' % s
    string = int(min(1, m)) * ('%dm ' % m) + string
    string = int(min(1, h)) * ('%dh ' % h) + string
    string = int(min(1, d)) * ('%dh ' % d) + string
    return string


def progress(current, total, start, last_update):
    """Output progress with progress bar"""
    # Don't update if less than a tenth of a second has passed
    if (time.time() - last_update) < 0.1 and not current == total:
        return last_update

    # Calculate time passed and time remaining
    passed = time.time() - start
    remaining = passed / float(current) * float(total - current)

    # Output progress
    sys.stdout.write('\r{0: >{1}}/{2} '.format(current, len(str(total)), total))
    # Output progress bar
    length = math.floor(30 * current / float(total))
    sys.stdout.write('[%s] - ' % ('=' * length + '>' * min(1, 30 - length) + '.' * (30 - length - 1)))

    # If task has completed then report time taken
    if current == total:
        sys.stdout.write(seconds_to_string(passed))
    # If task has not completed then report eta
    else:
        sys.stdout.write('ETA: ' + seconds_to_string(remaining))

    # Output padding
    sys.stdout.write(' ' * 20)
    # Allow progress bar to persist if it's complete
    if current == total:
        sys.stdout.write('\n')
    # Flush to standard out
    sys.stdout.flush()
    # Return the time of the progress update
    return time.time()


model = Sequential()
model.add(Dense(25, activation_function="relu", input_shape=2))
model.add(Dense(50, activation_function="relu"))
model.add(Dense(50, activation_function="relu"))
model.add(Dense(25, activation_function="relu"))
model.add(Dense(1, activation_function="sigmoid"))


def init_layers(seed = 2):
    # random seed initiation
    np.random.seed(seed)

    # Iterate over the layers of the neural network
    for layer in model.layers:
        # The number of units in a layer is equivilant to the number of outputs
        layer.W = np.random.randn(layer.units, layer.input_shape) * 0.1
        layer.b = np.random.randn(layer.units, 1) * 0.1


def forward_propagation(X):
    # The input X acts as the activation, A_prev, for the previous layer
    A_prev = X
    # Iterate over the layers of the neural network
    for layer in model.layers:
        # Calculate the affine transformation, Z, for the current layer
        Z = np.dot(layer.W, A_prev) + layer.b
        # Calculate the activations, A, for the current layer
        A = layer.activation_function(Z)
        # Update the pointer to the activations for the previous layer, A_prev, ready for the next iteration
        A_prev = A


    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr
        
        # extraction of the activation function for the current layer
        activ_function_curr = layer['activation']
        # extraction of W for the current layer
        W_curr = params_values['W' + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values['b' + str(layer_idx)]
        # calculation of activation for the current layer

        Z_curr = np.dot(W_curr, A_prev) + b_curr
        A_curr, Z_curr =  activation(Z_curr), Z_curr
        
        # saving calculated values in the memory
        memory['A' + str(idx)] = A_prev
        memory['Z' + str(layer_idx)] = Z_curr
       
    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory


def get_cost_value(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)


# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation='relu'):
    # number of examples
    m = A_prev.shape[1]
    
    # calculation of the activation function derivative
    dZ_curr = dA_curr * activation.derivative(Z_curr)
    
    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    
    # number of examples
    m = Y.shape[1]
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(Y_hat.shape)
    
    # initiation of gradient descent algorithm
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer['activation']
        
        dA_curr = dA_prev
        
        A_prev = memory['A' + str(layer_idx_prev)]
        Z_curr = memory['Z' + str(layer_idx_curr)]
        
        W_curr = params_values['W' + str(layer_idx_curr)]
        b_curr = params_values['b' + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values['dW' + str(layer_idx_curr)] = dW_curr
        grads_values['db' + str(layer_idx_curr)] = db_curr
    
    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):

    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values['W' + str(layer_idx)] -= learning_rate * grads_values['dW' + str(layer_idx)]        
        params_values['b' + str(layer_idx)] -= learning_rate * grads_values['db' + str(layer_idx)]

    return params_values;


def train(X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    # initiation of neural net parameters
    params_values = init_layers(nn_architecture)
    # initiation of lists storing the history 
    # of metrics calculated during the learning process 
    cost_history = []
    accuracy_history = []

    last_update = start = time.time()
    
    # performing calculations for subsequent iterations
    for i in range(epochs):
        last_update = progress(i + 1, epochs, start, last_update)

        # step forward
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        
        # calculating metrics and saving them in history
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        # updating model state
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
        if(i % 50 == 0):
            if(verbose):
                print('Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}'.format(i, cost, accuracy))
            if(callback is not None):
                callback(i, params_values)
            
    return params_values


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.1


X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)


# Training
params_values = train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), model, 10000, 0.01)


# Prediction
Y_test_hat, _ = full_forward_propagation(np.transpose(X_test), params_values, nn_architecture)

# Accuracy achieved on the test set
acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print('Test set accuracy: {:.2f} - David'.format(acc_test))