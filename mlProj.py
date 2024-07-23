import numpy as np


def nonlin(x, deriv = False):
    if deriv == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


#input dataset
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

#output dataset
y = np.array([[0, 0, 1, 1]]).T


np.random.seed(1)

syn0 = 2*np.random.random((3, 1)) - 1
for i in range(10000):

    #forward propogation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    #error
    l1_error = y - l1

    #multiplying error with slope of sigmoid function
    l1_delta = l1_error * nonlin(l1, True)

    #updated weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output after Training: ")
print(l1)