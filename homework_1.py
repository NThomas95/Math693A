'''
This is the code for the first homework assignment in M693A. The objective is to minimize the rosenbrock function, 
described by:

    f(x,y) = 100(y-x^2)^2 + (1-x)^2

We need to evaluate two methods: the steepest decent method and the newton direction method. We must use the 
backtracking algorithm to determine our step length. In addition, we need to evalue two different starting points,
[1.2, 1.2] and [-1.2, 1]. We also need to print the value of our step length at each iteration.

Author: Nathan Thomas

Instructor: Dr. Peter Blomgren
Course: Math693A Computational Optimization
Date: September 21, 2018, 12:00PM
'''


# imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# uncomment this section to see the plot of the function we are minimizing
# This is a quick plot of our function to minimize
X = np.arange(-2, 2, 0.01)  # establish x and y axis to plot
Y = np.arange(-2, 2, 0.01)
X, Y = np.meshgrid(X,Y)  # mesh them. Matplotlib needs this to plot 3d functions. Essentially form the x-y plane
Z = 100*(Y-X**2)**2 + (1-X)**2  # function values accross the x-y plane
fig = plt.figure()  # establish the figure you are plotting
ax = fig.gca(projection='3d')  # establish the plot is 3d
surf = ax.plot_surface(X, Y, Z)  # plot the function
plt.show()  # show the plot.


def rosenbrock_function(location):
    '''
    This function rurns the value of the rosenbrock function for all x,y
    
    param: location: Where are you currently in the x-y plane

    return: z: The value of the rosenbrock function at (x)
    '''

    z = 100*(location[1] - location[0]**2)**2 + (1-location[0])**2  # the value

    return z

def rosenbrock_derivative(location):
    '''
    This function returns the value of the gradiant of the rossenbrock function at specific points
    note that it does not symbolically differentiate. It returns a numeric 1-d vector of length 2

    param: location: Where are you currently located in the x-y plane

    return: d_location: The numerical value of the derivative at x
    '''

    df1 = -400*(location[1]-location[0]**2)*location[0] - 2*(1-location[0])
    df2 = 200*(location[1] - location[0]**2)

    d_location = np.array([df1, df2])

    return d_location

def rosenbrock_hessian(location):
    '''
    This function returns the hessian of the rosenbrock function at specific points. Note that it is not a symbolic
    hessian. It returns a 2x2 numeric matrix.

    param: location: Where is your current location in the x-y plane?

    return: H: The hessian matrix of the rosenbrock function, evaluated at x
    '''
    
    # These are the individual elements of the hessian
    H11 = 1200*location[0]**2 - 400*location[1]+ 2
    H12 = -400
    H21 = -400*location[0]
    H22 = 200

    H = np.array([[H11, H12],  # This is the hessian
                  [H21, H22]])

    return H

def direction_normalizer(direction):
    '''
    This function takes a given input vector and returns the vector
    pointed in the same direction with magnitude 1.

    param: direction: What is the direction?

    return: normalized_direction: A vector pointing in the same direction as p but with length 1
    '''
    
    normalized_direction = direction/np.linalg.norm(direction)

    return normalized_direction

def backtracker(location, step_direction, step_length=1, function_ratio=10**-4, compression_factor=1/2):
    '''
    This function returns an appropriate step length for our current iteration. It takes the following
    
    param: location:  The current (x,y) value. AKA where are you on the plane right now
    param: step_direction: What is the direction you are traveling in on the x-y plane
    param: step_length: What is your initially proposed step length
    param: function ratio: How close to the actual function do you need you decent to be
    param: compression_factor: If your step length fails, how much should we compress it

    return: step_length: The new step length that satisfies the Armijo condition
    '''
     
    # While the Armijo condition is not satisfied, keep compressing the step length
    while (rosenbrock_function(location + step_length*step_direction) > rosenbrock_function(location)
           + function_ratio*step_length*(rosenbrock_derivative(location) @ step_direction)):
        
        step_length = compression_factor * step_length

    return step_length


########################################################################################################################
##########################################    Steepest Decent Method    ################################################
########################################################################################################################

alpha_0 = 1  # initial step length
x_0 = np.array([1.2, 1.2])  # initial point (starting location)
x = x_0
steepest_decent_steps = []
steepest_decent_points = []  # establish some lists to keep track of our steps, points, and values
steepest_decent_values = []

# Here begins the steepest decent method
count = 1
stop = 20000
while np.linalg.norm(rosenbrock_function(x)) >= 10**-8 and np.linalg.norm(rosenbrock_derivative(x)) >= 10**-8:

    print("Steepest Decent Iteration: " + str(count), end="\r")  # print our current iteration
    
    # If we haven't converged after $stop iterations, stop the script. Something is probably wrong
    if count == stop:
        break
    else:
        pass

    p = -1*rosenbrock_derivative(x)  # this is the step direction
    alpha = backtracker(x, p, alpha_0)  # this finds our appropriate step length
    x = x + alpha*-1*rosenbrock_derivative(x)  # this adjusts our x point by moving a length alpha along the direction p
    steepest_decent_steps.append(alpha)  # add the step length to our list
    steepest_decent_points.append(x)  # add the new point to our list
    steepest_decent_values.append(rosenbrock_function(x))  # add the value of the function to our list
    count += 1

print(("\nThe number of steps for the steepest decent method, starting at " + str(x_0) + ", was: "
      + str(np.size(steepest_decent_steps))))
print("The steepest decent method, starting at " + str(x_0) + ", converges to: " + str(x) + "\n")


########################################################################################################################
##########################################     Newton Method     #######################################################
########################################################################################################################

alpha_0 = 1  # initial step length
x_0 = np.array([-1.2, 1])  # initial point (starting location)
x = x_0
newton_direction_steps = []
newton_direction_points = []  # establish some lists to keep track of our steps, points, and values
newton_direction_values = []

# Here begins the newton method
count = 1  # count our iterations
stop = 10000
while np.linalg.norm(rosenbrock_function(x)) >= 10**-8 and np.linalg.norm(rosenbrock_derivative(x)) >= 10**-8:
    
    print("Newton Direction Iteration: " + str(count), end="\r")  # print our current iteration

    # If we haven't converged after $stop iterations, stop the script. Something is probably wrong
    if count == stop:
        print("\nNo convergance!")
        print("The hessian eigenvalues are: " + str(np.linalg.eigvals(hess)))  # check for negative e-vals
        break
    else:
        pass

    hess = rosenbrock_hessian(x)  # compute the hessian
    p = -1*np.linalg.inv(hess) @ rosenbrock_derivative(x)  # this is the step direction
    p = direction_normalizer(p)  # this makes sure p has length 1
    alpha = backtracker(x, p, alpha_0)  # find an appropriate step length
    x = x + alpha*p  # move x a length alpha along the direction p
    newton_direction_steps.append(alpha)  # add the step length to our list
    newton_direction_points.append(x)  # add the point on to our list of points
    newton_direction_values.append(rosenbrock_function(x))  # add the function value to our list of values
    count += 1
    #print(x)  # testing
    #print(alpha)  # testing
    #print(rosenbrock_derivative(x))  # testing

print(("\nThe number of steps for the newton direction method, starting at " + str(x_0) + ", was: "
      + str(np.size(newton_direction_steps))))
print("The newton direction method, starting at " + str(x_0) + ", converges to: " + str(x) + "\n")
