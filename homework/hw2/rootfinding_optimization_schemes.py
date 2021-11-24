import numpy
import warnings

"""
Python routines for all the root finding and optimization methods included in the class notebook, 05_rootfinding_optimization.ipynb

Including:
* bisection
* newton
* secant
* parabolic_interpolation
* golden_section
"""

def bisection(f,a,b,tol = 1.e-6, max_steps=1000):
    """ uses bisection to isolate a root x of a function of a single variable f such that f(x) = 0.
        the root must exist within an initial bracket a < x < b
        returns when f(x) at the midpoint of the bracket < tol
    
    Parameters:
    -----------
    
    f: function of a single variable f(x) of type float
    a: float
        left bracket a < x
    b: float
        right bracket x < b
        
        Note:  the signs of f(a) and f(b) must be different to insure a bracket
    tol: float
        tolerance.  Returns when |f((a+b)/2)| < tol
    max_steps: int
        maximum number of iteration steps
        
    Returns:
    --------
    x: float
        midpoint of final bracket
    x_array: numpy array
        history of bracket centers (for plotting later)
        
    Raises:
    -------
    ValueError:  
        if initial bracket is invalid 
    Warning: 
        if number of iterations exceed MAX_STEPS
    
    """
    MAX_STEPS = max_steps
    
    # initialize
    delta_x = b - a
    c = a + delta_x / 2.0
    c_array = [ c ]
    
    f_a = f(a)
    f_b = f(b)
    f_c = f(c)
    
    # check bracket
    if numpy.sign(f_a) == numpy.sign(f_b):
        raise ValueError("no bracket: f(a) and f(b) must have different signs")
        
    # Loop until we reach the TOLERANCE or we take MAX_STEPS
    for step in range(1, MAX_STEPS + 1):
        # Check tolerance - Could also check the size of delta_x
        # We check this first as we have already initialized the values
        # in c and f_c
        if numpy.abs(f_c) < tol:
            break

        if numpy.sign(f_a) != numpy.sign(f_c):
            b = c
            f_b = f_c
        else:
            a = c
            f_a = f_c
        delta_x = b - a
        c = a + delta_x / 2.0
        f_c = f(c)
        c_array.append(c)
        
    if step == MAX_STEPS:
        warnings.warn('Maximum number of steps exceeded')
    
    return c, numpy.array(c_array)

def newton(f, f_prime, x0, tol=1.e-6, max_steps=200):
    """ uses newton's method to find a root x of a function of a single variable f
    
    Parameters:
    -----------
    f: function f(x)
        returns type: float
    f_prime: function f'(x)
        returns type: float
    x0: float
        initial guess
    tolerance: float
        Returns when |f(x)| < tol
    max_steps: int
        maximum number of iteration steps
        
    Returns:
    --------
    x: float
        final iterate
    x_array: numpy array
        history of iteration points
        
    Raises:
    -------
    Warning: 
        if number of iterations exceed MAX_STEPS
    """
    MAX_STEPS = max_steps
    
    x = x0
    x_array = [ x0 ]
    for k in range(1, MAX_STEPS + 1):
        x = x  - f(x) / f_prime(x)
        x_array.append(x)
        if numpy.abs(f(x)) < tol:
            break
        
    if k == MAX_STEPS:
        warnings.warn('Maximum number of steps exceeded')
    
    return x, numpy.array(x_array)

def secant(f, x0, x1, tol=1.e-6, max_steps=100):
    """ uses a linear secant method to find a root x of a function of a single variable f
    
    Parameters:
    -----------
    f: function f(x)
        returns type: float
    x0: float
        first point to initialize the algorithm
    x1: float
        second point to initialize the algorithm x1 != x0        
    tolerance: float
        Returns when |f(x)| < tol
    max_steps: int
        maximum number of iteration steps
        
    Returns:
    --------
    x: float
        final iterate
    x_array: numpy array
        history of iteration points
        
    Raises:
    -------
    ValueError:
        if x1 is too close to x0
    Warning: 
        if number of iterations exceed MAX_STEPS
    """
    MAX_STEPS = max_steps
    
    if numpy.isclose(x0, x1):
        raise ValueError('Initial points are too close (preferably should be a bracket)')
        
    x_array = [ x0, x1 ]
    for k in range(1, MAX_STEPS + 1):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x_array.append(x2)
        if numpy.abs(f(x2)) < tol:
            break
        x0 = x1
        x1 = x2
        
    if k == MAX_STEPS:
        warnings.warn('Maximum number of steps exceeded')
    
    return x2, numpy.array(x_array)

def parabolic_interpolation(f, bracket, tol=1.e-6, max_steps=100):
    """ uses repeated parabolic interpolation to refine a local minimum of a function f(x)
    this routine uses numpy functions polyfit and polyval to fit and evaluate the quadratics
    
    Parameters:
    -----------
    f: function f(x)
        returns type: float
    bracket: array
        array [x0, x1] containing an initial bracket that contains a minimum   
    tolerance: float
        Returns when relative error of last two iterates < tol 
    max_steps: int
        maximum number of iteration steps
        
    Returns:
    --------
    x: float
        final estimate of the minima
    x_array: numpy array
        history of iteration points
        
    Raises:
    -------
    Warning: 
        if number of iterations exceed MAX_STEPS
    """
    MAX_STEPS = max_steps
    
    x = numpy.zeros(3)
    x[:2] = bracket
    x[2] = (x[0] + x[1])/2.
        
    x_array = [ x[2] ]
    for k in range(1, MAX_STEPS + 1):
        poly = numpy.polyfit(x, f(x), 2)
        x[0] = x[1]
        x[1] = x[2]
        x[2] = -poly[1] / (2.0 * poly[0])
        x_array.append(x[2])
        if numpy.abs(x[2] - x[1]) / numpy.abs(x[2]) < tol:
            break
        
    if k == MAX_STEPS:
        warnings.warn('Maximum number of steps exceeded')
    
    return x[2], numpy.array(x_array)
