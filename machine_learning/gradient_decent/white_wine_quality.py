from gradient_decent.gradient_decent import gradient_descent
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datasets

# Load White Wine Quality Dataset
white_wine = datasets.load_dataset('codesignal/wine-quality', split='white')
white_wine = pd.DataFrame(white_wine)

# Only consider the 'density' column as a predictive feature for now
x = pd.DataFrame(white_wine['density'])
y = white_wine['quality']

# Splitting datasets into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# We set our parameters to start at 0
theta = np.zeros(x_train.shape[1]).reshape(-1, 1)

# Define the number of iterations and alpha value
alpha = 0.001
iters = 2000

# Applying Gradient Descent
y_train = np.array(y_train).reshape(-1, 1)
g, theta_list, cost_list = gradient_descent(x_train, y_train, theta, alpha, iters)


plt.plot(range(1, iters + 1), cost_list, color='green')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')
plt.show()
