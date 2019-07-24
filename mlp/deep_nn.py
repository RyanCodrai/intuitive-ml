import numpy as np
import time
import sys
import math

from models import Sequential
from layers import Dense, Input, Activation


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
model.add(Input(2))
model.add(Dense(25))
model.add(Activation("relu"))
model.add(Dense(50))
model.add(Activation("relu"))
model.add(Dense(50))
model.add(Activation("relu"))
model.add(Dense(25))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))


def initialise_layer_parameters(seed = 2):
    # random seed initiation
    np.random.seed(seed)

    # Iterate over the layers of the neural network
    for layer in model.layers[1:]:
        layer.initalise()


def get_cost_value(y_hat, y):
    # Calculate the number of training examples
    n = y_hat.shape[1]
    # Calculate the cross-entropy cost
    cost = -1 / n * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))
    # Return the cost
    return np.squeeze(cost)


# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(y_hat, y):
    y_hat_ = convert_prob_into_class(y_hat)
    return (y_hat_ == y).all(axis=0).mean()


def backward_propagation(y_hat, Y):
    # number of examples
    m = Y.shape[1]
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(y_hat.shape)
    # initiation of gradient descent algorithm
    loss_derivative = - (np.divide(Y, y_hat) - np.divide(1 - Y, 1 - y_hat));

    model.layers[-1].backward_propogation(loss_derivative)


def update(learning_rate):
    # Iterate over the layers of the neural network exclusing the input layer
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db


def train(X, Y, epochs, learning_rate, verbose=False, callback=None):
    # initiation of neural net parameters
    initialise_layer_parameters()
    # initiation of lists storing the history 
    # of metrics calculated during the learning process 
    cost_history = []
    accuracy_history = []

    last_update = start = time.time()
    
    # performing calculations for subsequent iterations
    for i in range(epochs):
        last_update = progress(i + 1, epochs, start, last_update)

        # step forward
        y_hat = model.layers[0].forward_propagation(X)
        # calculating metrics and saving them in history
        cost = get_cost_value(y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(y_hat, Y)
        accuracy_history.append(accuracy)
        
        # step backward - calculating gradient
        backward_propagation(y_hat, Y)
        # updating model state
        update(learning_rate)
        
        if(i % 50 == 0):
            if(verbose):
                print('Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}'.format(i, cost, accuracy))
            if(callback is not None):
                callback(i, params_values)


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.1


X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)


# Training
train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), 10000, 0.01)


# Prediction
Y_test_hat = model.layers[0].forward_propagation(np.transpose(X_test))

# Accuracy achieved on the test set
acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print('Test set accuracy: {:.2f} - David'.format(acc_test))