import numpy as np

def gradient_descent(x, y, theta, alpha, iterations):
    """
    x -- input dataset
    y -- target dataset
    theta -- initial parameters
    alpha -- learning rate
    iterations -- the number of times to execute the algorithm
    """

    m = y.size # number of data points
    cost_list = [] # list to store the cost function value at each iteration
    theta_list = [theta] # list to store the values of theta at each iteration

    for i in range(iterations):
        # calculate our prediction based on our current theta
        prediction = np.dot(x, theta)

        # compute the error between our prediction and the actual values
        error = prediction - y

        # calculate the cost function
        cost = 1 / (2*m) * np.dot(error.T, error)

        # append the cost to the cost_list
        cost_list.append(np.squeeze(cost))

        # calculate the gradient descent and update the theta
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))

        # append the updated theta to the theta_list
        theta_list.append(theta)

    # return the final values of theta, list of all theta, and list of all costs, respectively
    return theta, theta_list, cost_list
