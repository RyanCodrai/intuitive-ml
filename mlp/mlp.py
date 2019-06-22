import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
    """
    - Activation function that transform linear inputs to nonlinear outputs.
    - Bounds output between 0 and 1 inclusively so that it can be interpreted as a probability.
    """
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
W0 = 2*np.random.random((3,1)) - 1

for iter in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,W0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    W0 += np.dot(l0.T,l1_delta)

    print(l1)

print("Output After Training:")
print(l1)