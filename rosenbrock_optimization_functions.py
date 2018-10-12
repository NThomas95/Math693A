'''
This code is a function repository contraining functions used during my Math693A, Advanced Computational Optimization,
course at SDSU. 

Author: Nathan Thomas

Instructor: Dr. Peter Blomgren
Course: Math693A Computational Optimization
'''


# imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    
    
def phi(location, step_length, step_direction):
    '''
    This function returns the value of phi(alpha) = f(x+alpha*p). Phi is the rosenbrock function if we fix the location
    and step direction and then use the step length as our parameter. Double checked. This function is good.
    
    param: location: where are you in the (x,y) plane
    param: step_length: how far do you want to step
    param: step_direction: what direction to step
    
    return: value: the value of phi(alpha) for the given alpha
    '''
    
    x_diff = location[0] + step_length*step_direction[0]  # difference between x location and new x location
    y_diff = location[1] + step_length*step_direction[1]  # difference between y location and new y location
    
    value = 100*(y_diff - x_diff**2)**2 + (1 - x_diff)**2  # value of the rosenbrock function at new location
             
    return value
    
    
def dphi(location, step_length, step_direction):
    '''
    This function returns the derivative wrt alpha of the phi function directly above this one. Since phi is only a 
    function of alpha, this function returns a scalar. derivative was taken by hand and the coded. There might be an
    error.
    
    param: location: where are you in the (x,y) plane
    param: step_length: how far do you want to step
    param: step_direction: what direction to step
    
    return: value: the value of dphi(alpha) for the given alpha
    '''
    
    x_diff = location[0] + step_length*step_direction[0]  # difference between x location and new x location
    y_diff = location[1] + step_length*step_direction[1]  # difference between y location and new y location
    
    value = 200*(y_diff - x_diff**2)*(step_direction[1] - 2*x_diff*step_direction[0]) - 2*step_direction[0]*(1 - x_diff)
            
    return value
    
    
def rosenbrock_quadratic_interpolate(alpha_0, location, step_direction):
    '''
    This function will return the minimizer of a quadratic modeling of the rosebrock function. It takes the value of 
    the rosenbrock function at alpha=0 and the derivative at alpha=0 as well as the value at the current alpha and of 
    course alpha itself. It will return a new alpha which minimizes the quadratic interpolation.
    
    param: alpha_0: The first step length guess
    param: location: location in the (x,y) plane
    param: step_direction: what direction are we moving
    
    return: alpha_new: The minimizer of the quadratic interpolation
    '''

    dphi_alpha_0 = dphi(location, alpha_0, step_direction)
    phi_alpha_0 = phi(location, alpha_0, step_direction)
    dphi_0 = dphi(location, 0, step_direction)
    phi_0 = phi(location, 0, step_direction)
    print("dphi is: " + str(dphi_alpha_0))
    print("phi is: " + str(phi_alpha_0))
    
    alpha_new = -(dphi_0*alpha_0**2) / (2*dphi_alpha_0 - phi_0 - dphi_0*alpha_0)
                 
    print("alpha_new is: " + str(alpha_new))
    
    return alpha_new
    
    
def rosenbrock_cubic_interpolate(alpha_0, alpha_1, location, step_direction):
    '''
    This function will take information about where you are and what your previous guess for step length was to build
    a cubic model of the rosenbrock function and then find the minimizer. Details are in chapter 3 of Numerical 
    Optimization
    
    param: alpha_0: The first step length
    param: alpha_1: the step length generated on the previous interpolation. This may be quadratic or cubic
    param: location: where are you in the (x,y) plane
    param: step_direction: What direction are we moving. Steepest decent or Newton
    
    return: alpha_new: The minimizer of the cubic interpolation
    '''
    
    # Create some variables that make the calculation of the coefficients easier to see and debug. This formula is 
    # taken from the Numerical optimization, Jorge Nocedal, p 58
    mat1 = np.array([[alpha_0**2, -alpha_1**2],  # first matrix
                     [-alpha_0**3, alpha_1**3]])
    mat2 = np.array([[phi(location, alpha_1, step_direction) - phi(location, 0, step_direction) - dphi(location, 0, step_direction)*alpha_1],
                     [phi(location, alpha_0, step_direction) - phi(location, 0, step_direction) - dphi(location, 0, step_direction)*alpha_0]])
    leading_multiplier = 1 / (alpha_0**2 * alpha_1**2 * (alpha_1 - alpha_0))  # a leading constant
    coefficients = leading_multiplier * (mat1 @ mat2)  # actually calculate the coefficients
    print("coefficients are: " + str(coefficients))
    
    # this formula is also taken from p. 58. alpha_new is the minimizer of the cubic interpolation
    alpha_new = (-coefficients[1] + np.sqrt(coefficients[1]**2 - 3*coefficients[0]*dphi(location, 0, step_direction))) / (3*coefficients[0])
    
    return alpha_new
    
    
def zoom(alpha_lo, alpha_hi, interpolater="quadratic")
    '''
    This function will return a step length by building a quadratic model of the function and finding the min/max.
    
    param: phi_0: The initial value of the function. This might be phi(alpha_lo)
    param: dphi_0: The initial derivative of the function. The might be dphi(alpha_lo)
    param: phi_alpha_current: The phi at the most recently calculated alpha
    param: phi_alpha_previous: The phi at the previously calculated alpha
    
    return: alpha_star: The optimal step length
    '''
    
    if interpolater == "quadratic":  # If this is true we are trying to interpolate for the first time. Choose quadratic
        alpha_new = rosenbrock_quadratic_interpolate(alpha_0, location, step_direction)
        
    elif interpolater == "cubic":  # If we are here, the quadratic failed and we must try a cubic
        alpha_new = rosenbrock_cubic_interpolate(alpha_0, location, step_direction)
        
        
    phi_alpha_new = rosenbrock_function(location + alpha_new*step_direction)
    if phi_alpha_new > phi_0 + c_1*alpha_new*dphi_0 or phi_alpha_new >= phi_:
        alpha_hi = alpha_new
        
    else:
        dphi_alpha_new = rosenbrock_derivative(location + alpha_new*step_direction)
        if np.abs(dphi_alpha_new) <= -c_2*dphi_0:
            alpha_star = alpha_new
            return alpha_star
            
        if dphi_alpha_new*(alpha_hi - alpha_lo) >= 0:
            alpha_hi = alpha_lo
            
            
    count += 1
    

if __name__ == "__main__":
    alpha_6 = 0.1  # initial step length
    x_0 = np.array([1.2, 1.2])  # initial point (starting location)
    x = x_0    
    p = -1*rosenbrock_derivative(x)  # this is the step direction
    p = direction_normalizer(p)
    alpha_7 = rosenbrock_quadratic_interpolate(alpha_6, x_0, p)   # this finds our appropriate step length
    #alpha_8 = rosenbrock_cubic_interpolate(alpha_6, alpha_7, x_0, p) 
    print(alpha_7)
    #print(alpha_8)

            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    