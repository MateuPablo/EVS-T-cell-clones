import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from lmfit import Model, Parameters


"""
Data analysis of the power law exponents of clonotype distributions

Functions
=========
    exponent_fit : function to filter the repertoire and fit it to the power law distribution
    exponent_analysis : function to analyse the resulting power law exponents
"""


def log_G(logf, a, K):
    """
    This function is the log-log rank-frequency distribution to fit
    """
    
    return K - a*logf

def G(f, a, K):
    return 10**(K)*f**(-a)

def data_plot(file_name, patient_frame, frame_fit, fmax_discard, a, K):
    """ 
    This function creates a plot with the fitting results.
    """
    
    ## ax1 : full repertoire with printed cutoff and discarded region ##
        
    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (10.5,4.5))
    patient_name = file_name[0:file_name.index('.')]
        
    ax1.axvspan(0, fmax_discard, 0, 1, color='darkcyan', alpha=0.25, lw=0, label = 'discarded data')  
    ax1.step(patient_frame['frequency'],patient_frame['rank'], lw = 2, color = 'purple', label = 'patient data')
    ax1.plot([fmax_discard, fmax_discard], [0,ax1.get_ylim()[1]], color = 'black', linestyle = 'dashed', 
             lw = 2, label = 'selected threshold')
    ax1.set_xscale('log')
    ax1.set_yscale('log') 
    ax1.set_xlabel('Normalised clone size, f')
    ax1.set_ylabel('Rank')
    ax1.set_title('Patient ' + patient_name)
    ax1.legend() 
    
    
    ## ax2 : fitted data with fitting curve in green (a>=1) or red (a<1) ##

    f_plot = np.linspace(frame_fit['frequency'].min(), frame_fit['frequency'].max(), 5)
    ax2.step(frame_fit['frequency'], frame_fit['rank'], lw = 2, color = 'purple',
             label = 'non discarded patient data')
    if (a>1):
        ax2.plot(f_plot, G(f_plot, a, K), lw = 2, color = 'tab:green', 
                label = r'fit to power law')
    else:
        ax2.plot(f_plot, G(f_plot, a, K), lw = 2, color = 'tab:red', 
                 label = r'fit to power law')
    ax2.set_xscale('log')
    ax2.set_yscale('log') 
    ax2.set_xlabel('Normalised clone size')
    ax2.set_ylabel('Rank')
    ax2.set_title('Patient ' + patient_name)
    ax2.legend()
    
def exponent_fit(file_name, read_path):
    """
    This function receives a patient's dataset and does the following operations:
        filtering of data below the threshold
        fit to rank-frequency power law with estimation and analysis of exponent
        optional: plot of fitting results
    
    Parameters
    ==========
        file_name : string
            patient's dataset file name
        read_path : string
            path where the datasets are stored
     
    Output
    ======
        a : float
            exponent of the power law distribution
        r2 : float
            R² coefficient of the fit
        val : string
            validity of the fit: yes or no
    """    
    
    patient_frame = pd.read_csv(read_path + file_name, sep='\t', low_memory=False)

    
    ## Creating the rank-frequency data ##
    
    patient_frame  = patient_frame.sort_values(by = 'frequency', ascending = False, ignore_index= True)
    patient_frame['rank'] = patient_frame.index + 1
    patient_frame = patient_frame.sort_values(by='frequency', ascending = True, ignore_index = True)
        
      
    ## Discarding frequencies below the threshold ##
    
    C_t = 16
    fmax_discard = np.max(patient_frame['frequency'][patient_frame['count'] < C_t])
    frame_fit = patient_frame[patient_frame['frequency'] >= fmax_discard]
        
        
    ## Fit to log-log rank-frequency distribution ##
    
    plaw_model = Model(log_G)  # defining the model and parameters of the fit
    param = Parameters()
    param.add('K', value = -8)
    param.add('a', value = 5, min = 0)
     
    logf_fit = np.log10(frame_fit['frequency'])  # data points and fit
    logG_fit = np.log10(frame_fit['rank'])
    res = plaw_model.fit(logG_fit, logf = logf_fit, a = param['a'], K = param['K'])
    
    a = res.values['a']  # parameters and estimates from the fit
    K = res.values['K']
    r2 = res.rsquared   
    if (a < 1 or r2 < 0.95):
        val = 'no'
    else:
        val = 'yes'

        
    ## Optional: plot of fitting results ##
    
    plt.close()
    data_plot(file_name, patient_frame, frame_fit, fmax_discard, a, K)
    
    return a, res.rsquared, val
        
def exponent_analysis(fit_results):
    """
    This function receives the results from the power law fits
    and prepares the data for its fit to a distribution
    
    Parameter
    ==========
        fit_results : pandas DataFrame
            contains three columns with the a, R² and validity results from each fit
     
    Output
    ======
        analysis_results : pandas DataFrame
            contains two columns with the a and b values for the valid fits
    """
    
    
    analysis_results = fit_results[fit_results['valid'] == 'yes']  # remove the non-valid fits
    del analysis_results['r2']
    del analysis_results['valid']
    
    analysis_results.insert(1, 'b', np.log10(analysis_results['a']-1), True)  # create the b = log10(a-1) data
    
    return analysis_results
    
    