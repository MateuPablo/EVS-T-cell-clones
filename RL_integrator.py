from __future__ import print_function
import numpy as np
from numba import jit

""" 
This code is an extract from the full differint package
It includes some slight modification of the RL integrator:
    **args : the function is modified to allow for arguments
    jit : just in time compilation is introduced to speed up the matrix operations
    RL outcome : the result is no longer an array with evaluationt, but a simple float with the result
The comments from the original code are mostly respected
"""


def isInteger(n):
    if n.imag:
        return False
    if float(n.real).is_integer():
        return True
    else:
        return False

def isPositiveInteger(n):
    return isInteger(n) and n > 0

def checkValues(alpha, domain_start, domain_end, num_points, support_complex_alpha=False):
    """ Type checking for valid inputs. """
    
    assert isPositiveInteger(num_points), "num_points is not an integer: %r" % num_points
    
    assert isinstance(domain_start, (int, np.integer, float, np.floating)),\
                     "domain_start must be integer or float: %r" % domain_start
        
    assert isinstance(domain_end, (int, np.integer, float, np.floating)),\
                     "domain_end must be integer or float: %r" % domain_end

    if not support_complex_alpha:
        assert not isinstance(alpha, complex), "Complex alpha not supported for this algorithm."
    
    return   

def functionCheck(f_name, domain_start, domain_end, num_points, **args):
    """ Check if function is callable and assign function values. """
    
    # Define the function domain and obtain function values.
    if hasattr(f_name, '__call__'):
        # If f_name is callable, call it and save to a list.
        x = np.linspace(domain_start, domain_end, num_points)
        f_values = list(map(lambda t: f_name(t, **args), x)) 
        step_size = x[1] - x[0]
    else:
        num_points = np.size(f_name)
        f_values = f_name
        step_size = (domain_end - domain_start)/(num_points-1)
    return f_values, step_size

def Gamma(z):
    """ Paul Godfrey's Gamma function implementation valid for z complex.
        This is converted from Godfrey's Gamma.m Matlab file available at
        https://www.mathworks.com/matlabcentral/fileexchange/3572-gamma.
        15 significant digits of accuracy for real z and 13
        significant digits for other values.
    """
    if not (type(z) == type(1+1j)):
        if isPositiveInteger(-1 * z):
            return np.inf
        from math import gamma
        return gamma(z)

    siz = np.size(z)
    zz = z
    f = np.zeros(2,)
        
    # Find negative real parts of z and make them positive.
    if type(z) == 'complex':
        Z = [z.real,z.imag]
        if Z[0] < 0:
            Z[0]  = -Z[0]
            z = np.asarray(Z)
            z = z.astype(complex)
    
    g = 607/128.
    
    c = [0.99999999999999709182,\
          57.156235665862923517,\
         -59.597960355475491248,\
          14.136097974741747174,\
        -0.49191381609762019978,\
        .33994649984811888699e-4,\
        .46523628927048575665e-4,\
       -.98374475304879564677e-4,\
        .15808870322491248884e-3,\
       -.21026444172410488319e-3,\
        .21743961811521264320e-3,\
       -.16431810653676389022e-3,\
        .84418223983852743293e-4,\
       -.26190838401581408670e-4,\
        .36899182659531622704e-5]
    
    if z == 0 or z == 1:
        return 1.
    
    # Adjust for negative poles.
    if (np.round(zz) == zz) and (zz.imag == 0) and (zz.real <= 0):
        return np.inf
        
    z = z - 1
    zh = z + 0.5
    zgh = zh + g
    
    # Trick for avoiding floating-point overflow above z = 141.
    zp = zgh**(zh*0.5)
    ss = 0.
    
    for pp in range(len(c)-1,0,-1):
        ss += c[pp]/(z+pp)
        
    sq2pi =  2.5066282746310005024157652848110;
    f = (sq2pi*(c[0]+ss))*((zp*np.exp(-zgh))*zp)
    
    # Adjust for negative real parts.
    #if zz.real < 0:
    #    F = [f.real,f.imag]
    #    F[0] = -np.pi/(zz.real*F[0]*np.sin(np.pi*zz.real))
    #    f = np.asarray(F)
    #    f = f.astype(complex)
    
    if type(zz) == 'complex':
        return f.astype(complex)
    elif isPositiveInteger(zz):
        f = np.round(f)
        return f.astype(int)
    else:
        return f

@jit(nopython = True) # jit compilation speeds up the matrix operations
def RLcoeffs(index_k, index_j, alpha):
    """Calculates coefficients for the RL differintegral operator.
    
    see Baleanu, D., Diethelm, K., Scalas, E., and Trujillo, J.J. (2012). Fractional
        Calculus: Models and Numerical Methods. World Scientific.
    """
    
    if index_j == 0:
        return ((index_k-1)**(1-alpha)-(index_k+alpha-1)*index_k**-alpha)
    elif index_j == index_k:
        return 1
    else:
        return ((index_k-index_j+1)**(1-alpha)+(index_k-index_j-1)**(1-alpha)-2*(index_k-index_j)**(1-alpha))

@jit(nopython = True)
def coeffMatrix(alpha,N):
    coeffMatrix = np.zeros((N,N))  ## this part acccelerates with numba, but not the whole function
    for i in range(N):
        for j in range(i):
            coeffMatrix[i,j] = RLcoeffs(i,j,alpha)
    np.fill_diagonal(coeffMatrix,1)

    return coeffMatrix
    
def RLmatrix(alpha, N):
    # Place 1 on the main diagonal.
    return coeffMatrix(alpha,N)/Gamma(2-alpha)

def RL(alpha, f_name, domain_start = 0.0, domain_end = 1.0, num_points = 100, **args):
    """ Calculate the RL algorithm using a trapezoid rule over 
        an array of function values.
        
    Parameters
    ==========
        alpha : float
            The order of the differintegral to be computed. For integration: -1
        f_name : function handle, lambda function, list, or 1d-array of 
                 function values
            This is the function that is to be differintegrated.
        domain_start : float
            The left-endpoint of the function domain. Default value is 0.
        domain_end : float
            The right-endpoint of the function domain; the point at which the 
            differintegral is being evaluated. Default value is 1.
        num_points : integer
            The number of points in the domain. Default value is 100.
        **args : float
            The parameters of the function. For instance, if f = f(x; a, b): **args ~ a = a0, b = b0.
            
    Output
    ======
        RL : float number
            The result of the differeintegral.
    
    Examples:
        >>> RL_sqrt = RL(-1, lambda x: np.sqrt(x))
        >>> RL_poly = RL(-1, f, 0., 1., 100, a = 2, b = 3), where f(x; a, b) = a*x**2 - b
    """
    
    # Flip the domain limits if they are in the wrong order.
    if domain_start > domain_end:
        domain_start, domain_end = domain_end, domain_start
    
    # Check inputs.
    checkValues(alpha, domain_start, domain_end, num_points)
    f_values, step_size = functionCheck(f_name, domain_start, domain_end, num_points, **args)
    
    # Calculate the RL differintegral.
    D = RLmatrix(alpha, num_points)
    RL = step_size**-alpha*np.dot(D, f_values)
    RL = RL[len(RL)-1]-RL[0]
    return RL
