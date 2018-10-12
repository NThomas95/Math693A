'''
This is the code for the first homework assignment in M693A. The objective is to minimize the rosenbrock function, 
described by:

    f(x,y) = 100(y-x^2)^2 + (1-x)^2

We need to evaluate two methods: the steepest decent method and the newton direction method. The difference between this
homework and homework 1 is that we must use the zoom algorithm instead of the backtracking line search. It would be very
helpful to modularize this code.

Author: Nathan Thomas

Instructor: Dr. Peter Blomgren
Course: Math693A Computational Optimization
Date: October 5, 2018, 12:00PM
'''


import rosenbrock_optimization_functions as rof
import numpy


print("Homework #2")
########################################################################################################################
##########################################    Steepest Decent Method    ################################################
########################################################################################################################
alpha_0 = 0  # initial step length
a_max = 0.5  # max value for the step length
alpha = 0.3  # starting guess for the step length

x_0 = np.array([1.2, 1.2])  # initial point (starting location)
x = x_0
steepest_decent_step_lengths = []
steepest_decent_points = []  # establish some lists to keep track of our steps, points, and values
steepest_decent_values = []
c1 = 10**-4  # armijo condition constant
c2 = 0.9  # curvature condition constant

# Here begins the steepest decent method
count = 1
stop = 20000
i = 0  # for indexing
while np.linalg.norm(rosenbrock_function(x)) >= 10**-8 and np.linalg.norm(rosenbrock_derivative(x)) >= 10**-8:

    print("Steepest Decent Iteration: " + str(count), end="\r")  # print our current iteration
    
    # If we haven't converged after $stop iterations, stop the script. Something is probably wrong
    if count == stop:
        break
    else:
        pass
        
        
    Algorithm goes here
    phi_alpha_i_1 = rof.phi(
    p = -1*rof.rosenbrock_derivative(x)  # this is the step direction
    phi_0 = rof.phi(x, 0, p)  # the phi function if we don't take a step
    phi_alpha_i = rof.phi(x, alpha, p)  # the value of the phi function at our location x, step length alpha, directionp
    dphi_0 = rof.dphi(x, 0, p)  # the value of the dphi function at our location x, step length alpha, and direction p
    
    
    if (phi_alphi_i > c1 * phi_o + c1*alpha*dphi_0) or (phi_alpha_i >= phi_alpha_i_1 and i+1 >1)
        
        
        
        
print(("\nThe number of steps for the steepest decent method, starting at " + str(x_0) + ", was: "
      + str(np.size(steepest_decent_steps))))
print("The steepest decent method, starting at " + str(x_0) + ", converges to: " + str(x) + "\n")


########################################################################################################################
##########################################     Newton Method     #######################################################
########################################################################################################################

alpha_0 = 1  # initial step length
x_0 = np.array([-1.2, 1])  # initial point (starting location)
newton_direction_step_lengths = []
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
        
        
        
    Algorithm goes here
        
        
print(("\nThe number of steps for the newton direction method, starting at " + str(x_0) + ", was: "
      + str(np.size(newton_direction_steps))))
print("The newton direction method, starting at " + str(x_0) + ", converges to: " + str(x) + "\n")
