import numpy as np
from scipy.integrate import quad
from scipy.integrate import fixed_quad
from RL_integrator import RL


""" 
Numerical implementation of the analytical distribution for the top clones

Functions
=========
    Standard extreme value statistics functions (s)
    Normalised extreme value statistics functions (n)
    Mixture extreme value statistics functions (m)
"""
    
def N(z, y):
    """ Diversity of repertoire as a function of alpha and f_min """
    
    return -10**(z)/(1+10**(z))*pow(10, -(1+10**(z))*y)*(1-pow(10, -10**(z)*y))**(-1)


""" 
Standard extreme value statistics functions

Parameters
==========
    x : float
       Top clone frequencies in log scale, log10(m)
    z : float
       Power law exponents in log scale, log10(alpha-1)
    y : float
       Minimum frequency in log scale, log10(f_min)
"""

def G_s(x, z, y):
    """ 
    Standard CDF 
    """
    
    return np.exp(-N(z, y)*pow(10, -(x-y)*(1+10**z)))
                  
def g_s(x, z, y):
    """ 
    Standard PDF 
    """
    
    return np.log(10)*np.exp(-N(z, y)*pow(10, (y-x)*(1+10**z)))*N(z, y)*(1+10**z)*pow(10, (y-x)*(1+10**z))


""" 
Normalised extreme value statistics functions

Parameters
==========
    f : float
        Clonotype frequency
    mu : float
        Fourier integration variable
    s : integer
        Number of divisions of the frequency integration interval
"""

def real_exp(f, mu, z, y):
    return (1+10**(z))*pow(10, y*(1+10**(z)))*np.cos(mu*f)*f**(-10**z-2)

def imag_exp(f, mu, z, y):
    return (1+10**(z))*pow(10, y*(1+10**(z)))*np.sin(mu*f)*f**(-2-10**z)

def avg_exponential(mu, x, z, y, s = 10):
    """ 
    Integral over the clonotype frequencies   
    """
    
    points = np.logspace(y, x, s)
    results = [quad(real_exp,points[i], points[i+1], args=(mu, z, y), epsabs=1e-06, 
                    epsrel=1e-06, limit= 200)[0] + 
               1j*quad(imag_exp, points[i], points[i+1], args=(mu, z, y), epsabs=1e-06, 
                       epsrel=1e-06, limit= 200)[0] for i in range(len(points)-1)]
    
    return np.sum(results)

def f0(x, z, y): 
    """ 
    Function <1;m> 
    """
    
    return 1-(10**(x-y))**(-1-10**z)

def exponential(mu, x, z, y, s = 10):
    return np.exp(N(z, y)*np.log(np.abs(avg_exponential(mu, x, z, y))/f0(x, z, y)))

def cosine(mu, x, z, y, s = 10): 
    return np.cos(N(z, y)*np.angle(avg_exponential(mu, x, z, y)/f0(x, z, y))-mu)

def I(mu, x, z, y, s = 10):
    I_res = [exponential(mu0, x, z, y)*cosine(mu0, x, z, y) for mu0 in mu]
    return I_res

def H(x, z, y, s = 10):
    """ 
    Correlated term of the CDF in model with normalisation 
    """
    
    return 2*(fixed_quad(I, 0, 2000,args = (x, z, y), n = 50)[0])

def Z(z, y, s = 10):
    """ 
    Partition function (2pi factor absorbed) 
    """
    
    return np.exp(-N(z, y)*10**((1+10**z)*y))*H(0, z, y)

def r_exponential(mu, x, z, y, s = 10):
    return np.exp((N(z, y)-1)*np.log(np.abs(avg_exponential(mu, x, z, y))/f0(x, z, y)))

def r_cosine(mu, x, z, y, s = 10):
    return np.cos((N(z, y)-1)*np.angle(avg_exponential(mu, x, z, y)/f0(x, z, y))-mu*(1 - 10**x))

def J(mu, x, z, y, s = 10):
    """ 
    Real integrand of the correlated term in the PDF in model
    with normalisation 
    """
    
    J_res = [r_exponential(mu0, x, z, y)*r_cosine(mu0, x, z, y) for mu0 in mu] 
    return J_res

def h(x, z, y, s = 10):
    """ 
    Correlated term of the PDF in model with normalisation 
    """
    
    return 2*(fixed_quad(J, 0, 2000, args = (x, z, y), n = 50)[0])

def G_n(x, z, y):
    """ 
    Normalised CDF 
    """
    
    return G_s(x, z, y)*H(x, z, y)/Z(z, y)

def g_n(x, z, y):
    """ 
    Normalised PDF 
    """
    
    return g_s(x, z, y)*h(x, z, y)/(Z(z, y)*f0(x, z, y))


""" 
Mixture extreme value statistics functions

Parameters
==========
    z_avg : float
        Average value of the variable z
    c : float
        Variance of the variable z in log space, c = log10(sigma²)
"""


def rho(z, z_avg, c):
    """ 
    Gaussian probability density function of the power law exponents
    """
    
    return 1/np.sqrt(2*np.pi)*10**(-c/2)*np.exp(-0.5*(z-z_avg)**2*10**(-c))

def integrand_G_m(z, x, z_avg, c, y):
    return rho(z, z_avg, c)*G_n(z, x, y)

def integrand_g_m(z, x, z_avg, c, y):
    return rho(z, z_avg, c)*g_n(x, z, y)

def G_m(x, z_avg, c, y):
    """ 
    Mixture CDF 
    """
    
    z_bounds = [-5, np.log10(15)]    
    return RL(-1, integrand_G_m, z_bounds[0], z_bounds[1], x = x, z_avg = z_avg, c = c, y = y)

def g_m(x,z_avg,c,y):
    """ 
    Mixture PDF 
    """
    
    z_bounds = [-5, np.log10(15)]
    return RL(-1, integrand_g_m, z_bounds[0], z_bounds[1], x = x, z_avg = z_avg, c = c, y = y)