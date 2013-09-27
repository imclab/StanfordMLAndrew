'''
Created on Sep 26, 2013

@author: ray
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # MATLAB-like plotting framework
   
"""
    Implements the univariate linear regression, as knowns as single variable
    linear regression, using gradient descent algorithm.
    
    Notices:
        1. Pandas implicitly aligns on the index   
"""

def computeCost(X, Y, theta):
    """
    COMPUTECOST Compute cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y    
    """
    
    # Number of training examples
    m = len(Y)
    
    # Octave Code
    # X = [ones(m, 1), X]
    # cost = (0.5/m).*(X*theta - Y)'*(X*theta - Y)
    
    predictions = X['population'].apply(lambda x : theta[0] + theta[1]*x)
    sumOfSquareErrors = (predictions - Y["profit"]).apply(lambda x : pow(x, 2)).sum()
    cost = (0.5/m)*sumOfSquareErrors
    
    #predictions = X['population'].apply(lambda x : theta[0] + theta[1]*x)
    #errors = predictions - Y['profit']
    #cost = (0.5/m)*errors*errors    

    return cost

def gradientDescent(X, Y, theta, alpha, num_iters):
    """ 
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha    
    """
    # Number of training examples
    m = len(Y)
    cost_history = []
    
    for num_iter in xrange(num_iters):
        predictions = X['population'].apply(lambda x : theta[0] + theta[1]*x)
        
        theta[0] = theta[0] - (alpha/m)*(predictions - Y['profit']).sum()
        theta[1] = theta[1] - (alpha/m)*((predictions - Y['profit'])*X['population']).sum()
        
        cost = computeCost(X, Y, theta)
        cost_history.append(cost)
        
    return theta, cost_history

if __name__ == '__main__':
    # Read CSV data
    data= pd.read_csv("ex1data1.txt",
                      names=["population", "profit"],
                      )
    X = pd.DataFrame(data['population'])
    Y = pd.DataFrame(data['profit'])
    m = len(Y)  # number of training examples
    
    #X = pd.DataFrame(data[1], np.ones(m))
    theta = pd.Series(np.zeros(2))   # Initialize fitting parameters
    
    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01;
    
    # Compute and display initial cost
    cost = computeCost(X, Y, theta)
    print('Initial cost: {0}'.format(cost))
    
    # Run gradient descent
    theta, cost_history = gradientDescent(X, Y, theta, alpha, iterations)
    print('Theta found by gradient descent: {0} {1}'.format(theta[0], theta[1]))
    
    # Plot the linear fit
    f = plt.figure()
    p = plt.scatter(data['population'], data['profit'], s=15, marker='x', label='Training Data')
    p.axes.set_title('Profit versus Population')
    p.axes.xaxis.label.set_text('Population in 10K')
    p.axes.yaxis.label.set_text('Profit in $10K')
    
    linear_fit_fun = lambda x : theta[0] + theta[1]*x
    x_seq = np.arange(4, 24, 0.1)
    y_seq = linear_fit_fun(x_seq)
    plt.plot(x_seq, y_seq, label='Univariate Linear Regression', color='red', 
             linestyle='solid', linewidth=2)
    plt.legend(loc="upper left")
    plt.savefig('linear_fit.pdf')
    
    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.dot([1, 3.5], theta)*10000
    print('For population = 35,000, we predict a profit of {0}'.format(predict1))
    
    predict2 = np.dot([1, 7.0], theta)*10000
    print('For population = 70,000, we predict a profit of {0}'.format(predict2))