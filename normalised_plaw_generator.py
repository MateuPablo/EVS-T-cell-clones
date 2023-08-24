import numpy as np
from scipy import fft
import pickle

"""
This code generates sequences of clonotype frequencies naturally normalised.
The normalisation is introduced in a rigid way during the generation process.
It emulates numerically the theoretical implementation via a delta function.

Main functions
==============
    generate_plaw_dist : function to generate plaw frequency samples
    prob_sumf : function to generate the probability density of the sum
                of frequencies
    constrained_dist_sampler : class to contrained distribution sampler
        run : function to generate the final normalised samples
"""


def generate_plaw_dist(alpha, f_min, n_freqs):
    """
    This function generates n_freq equi-spaced frequencies from f_min to 1
    that follow a power law distribution with characteristic exponent alpha
    
    Parameters
    ==========
        alpha : float
            universal exponent of the power law distributions
        f_min : float
            minimum frequency of the clonotype sample
        n_freqs : float 1d array
            number of frequencies contained in the sample
    
    Output
    ======
        freqs : float 1d array
            generated frequencies
        probs : float 1d array
            probability density of the frequencies (power law)
    """
    
    freqs = np.linspace(f_min, 1, n_freqs)
    weights = freqs**(-1-alpha)
    norm = weights.sum()
    probs = weights / norm
    return freqs, probs


def prob_sumf(probs, freqs, U, f_av = None, f_var = None):
    """
    This function returns the density probability of the summation of U 
    variables that follow probs distribution having freqs as values.
    It uses fast fourier transform to make the convolution.
    Specify average and variance of the frequencies to avoid their
    re-computation
    
    Parameters
    ==========
        probs : float 1d array
            probability density of the frequencies
        freqs : float 1d array
            sequence of frequencies
        U : integer
            diversity of the sequence (should be 1/f_av in our model)
        f_av : float
            average value of the frequency sequence
        f_var : float
            variance of the frequency sequence
    """
    
    if f_av == None:
        f_av = np.sum(probs*freqs)
    if f_var == None:
        f_var = np.sum(probs*freqs**2) - f_av**2
    f_min, df = freqs[0], freqs[1]-freqs[0]
    
    # Estimating the maximal frequency by usng gaussian approx
    max_f_est = max(1, U*f_av + 10*np.sqrt(U*f_var))
    
    # Extending the domain of the single prob dist until the max by inserting zeros
    N_freqs = int(np.round((max_f_est - f_min) / df) + 1)
    new_freqs = np.linspace(f_min, max_f_est, N_freqs)
    new_probs = np.append(probs, np.zeros(N_freqs - len(probs)))
    
    # Computing fft and convolution. The new frequencies are shifted by f_min
    f_ft = fft.fft(new_probs)
    return new_freqs + (U-1)*f_min, np.real(fft.ifft(f_ft**U))/df


class constrained_dist_sampler:
    """
    This class creates the sampler for normalised sequences. 
    Send the parameters of the problem as parameters (see example code).
    Contains the functions needed to generate the final samples.
    
    """
    
    def __init__(self, xs, probs, n_throws, init_R, sum_contraint = 1, final_dx = None):
        self.xs = xs
        self.ps = probs
        self.U = n_throws
        self.constr = sum_contraint
        self.init_R = init_R
        if final_dx == None:
            self.final_dx = xs[1] - xs[0]
        else:
            self.final_dx = final_dx
            
    def run(self, n_throws_steps, n_chunks):
        """
        It generates the samples of normalised frequencies.
        
        Parameters
        ==========
            self : parameters sent to the class
            n_throws_steps: integer 1d array
                checkpoints to discard sequences that cannot reach one
            n_chunks : integer
                chuncks into which the realisation is divided
        
        Output
        ======
            good_samples : float 2d array
                normalised frequency samples (each row is one sample) 
        """
        
        ## Building the probabilities of remaining xsum thorugh convolution ##
        
        self.xsum, self.p_xsum = [], []
        for k in n_throws_steps:
            xsum, p_xsum = prob_sumf(self.ps, self.xs, self.U-k)
            self.xsum.append(xsum)
            self.p_xsum.append(p_xsum)
            
        self.good_samples = np.empty((0, self.U))
        self.fract_good_samples = []
        R = int(self.init_R / float(n_chunks))
        
        
        ## At each chunk of realizations new good samples are generated ##
        
        for k in range(n_chunks):
            new_samples, fract_good_samples = self._run_one_chunk(n_throws_steps, R)
            self.good_samples = np.row_stack((self.good_samples, new_samples))
            self.fract_good_samples.append(fract_good_samples)
            print('Chunk:', k+1, 'n samples:', len(new_samples))
            
        self.fract_good_samples = np.array(self.fract_good_samples)
        return self.good_samples
            
    
    def _run_one_chunk(self, n_throws_steps, R):
        
        last_k = 0
        samples = np.empty((R, 0))
        fract_good_samples = []
        for ik, k in enumerate(n_throws_steps):
            n_steps = k - last_k
            new_samples = np.random.choice(self.xs, size=(R, n_steps), p=self.ps)
            samples = np.column_stack((samples, new_samples))
            samples = self._filter_unlikely_samples(samples, ik)
            fract_good_samples.append(len(samples)/R)
            samples = self._resample(samples, R)
            last_k = k
            
        new_samples = np.random.choice(self.xs, size=(R, self.U - last_k), p=self.ps)
        samples = np.column_stack((samples, new_samples))
        self.sum_x = samples.sum(axis=1)
        good_sample_mask = np.logical_and(self.sum_x > 1 - self.final_dx/2, self.sum_x < 1 + self.final_dx/2)
        return samples[good_sample_mask], fract_good_samples
    
    
    def _filter_unlikely_samples(self, samples, remaining_steps_i):

        ## Remaining distance to one ##
        
        xsum_left = 1 - samples.sum(axis=1)

        
        ## Prob of covering the remaining distance ##
        
        xsum, p_xsum = self.xsum[remaining_steps_i], self.p_xsum[remaining_steps_i]

        
        ## Computing which remaining sum_frequencies can reach 1 ##
        
        likely_xsum_mask = p_xsum*self.final_dx > 1/self.init_R
        if False not in likely_xsum_mask:
            min_xsum_i = 0
        else:
            min_xsum_i = list(likely_xsum_mask).index(True)
        min_xsum = xsum[min_xsum_i]
            
        if False not in likely_xsum_mask[min_xsum_i:]:
            max_xsum_i = -1
        else:
            max_xsum_i = list(likely_xsum_mask[min_xsum_i:]).index(False) + min_xsum_i
        max_xsum = xsum[max_xsum_i]

        
        ## Filtering the unlikely samples ##
        
        good_xsum_mask = np.logical_and(xsum_left > min_xsum, xsum_left < max_xsum)
        return samples[good_xsum_mask]

    
    def _resample(self, samples, R):
        n_new_samples = R - len(samples)
        resample_ind = np.random.choice(np.arange(len(samples)), size=n_new_samples)
        new_samples = np.take(samples, resample_ind, axis=0)
        return np.row_stack((samples, new_samples))
    
    
    
def sample_naive(R, U, freqs, probs, df):
    samples = np.random.choice(freqs, size=(R,U), p=probs)
    sum_samples = samples.sum(axis=1)
    samples_good_up = samples[sum_samples > 1 - df/2]
    sum_samples = samples_good_up.sum(axis=1)
    return samples_good_up[sum_samples < 1 + df/2]




def plot_sample_comparison(samples_naive, samples, ax1, ax2, ax3):
    
    av_fs_naive = np.mean(samples_naive, axis=0)
    av_fs = np.mean(samples, axis=0)
    ax1.set_xlabel('i', fontsize=12)
    ax1.set_ylabel(r'$\langle f_i \rangle$', fontsize=12)
    ax1.plot(av_fs_naive, label='naive')
    ax1.plot(av_fs, label='conditioned')

    corr1_fs_naive = np.mean(samples_naive[:,0,np.newaxis] * samples_naive, axis=0)
    corr1_fs = np.mean(samples[:,0,np.newaxis] * samples, axis=0)
    ax2.set_yscale('log')
    ax2.set_xlabel('i', fontsize=12)
    ax2.set_ylabel(r'$\langle f_1, f_i \rangle$', fontsize=12)
    ax2.plot(corr1_fs_naive, label='naive')
    ax2.plot(corr1_fs, label='conditioned')

    maxs_naive = np.max(samples_naive, axis=1)
    maxs = np.max(samples, axis=1)
    ax3.set_xlabel('max f', fontsize=12)
    ax3.set_ylabel('pdf', fontsize=12)
    ax3.hist(maxs_naive, bins=30, density=True, histtype='step', lw=2)
    ax3.hist(maxs, bins=30, density=True, histtype='step', lw=2)
    
    return ax1, ax2, ax3


def save_samples(directory, name, samples, alpha, f_min, n_freqs, df):
    
    f = open(directory+'/'+name+'.pickle', 'wb')
    pickle.dump(samples, f)
    f.close()
    
    f = open(directory+'/'+name+'_pars.txt', 'w')
    f.write('alpha\t'+str(alpha)+'\n')
    f.write('f_min\t'+str(f_min)+'\n')
    f.write('n_freqs\t'+str(n_freqs)+'\n')
    f.write('final_df\t'+str(df)+'\n')
    f.close()
    
    
def load_samples(directory, name):
    
    f = open(directory+'/'+name+'.pickle', 'rb')
    samples = pickle.load(f)
    f.close()
    
    par_dict = dict()
    f = open(directory+'/'+name+'_pars.txt', 'r')
    par_dict['alpha'] = float(f.readline().split('\t')[1])
    par_dict['f_min'] = float(f.readline().split('\t')[1])
    par_dict['n_freqs'] = float(f.readline().split('\t')[1])
    par_dict['final_df'] = float(f.readline().split('\t')[1])
    par_dict['U'] = len(samples[0])
    f.close()
    
    return samples, par_dict