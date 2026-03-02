
import numpy as np
import warnings
from jjmodel.mwdisk import local_run
from helpers import extract_model_tables


def log_prior_gaussian(theta, mu, sigma, lower=None, upper=None):
    r""" 
    Log Gaussian prior calculated as a sum over all parameter contributions. 

    Parameters:
    -----------
    - theta: list of floats
        Parameter values. 
    - mu: list of floats
        Means. 
    - sigma: list of floats
        Standard deviations. 

    Returns:
    --------
    - float
        Summed log Gaussian prior.  
    """
    # theta, mu, sigma: 1D arrays, same length
    if lower is not None and np.any(theta < lower): 
        return -np.inf
    if upper is not None and np.any(theta > upper): 
        return -np.inf
    z = (theta - mu) / sigma
    # dot product performs summing over parameters
    return -0.5 * np.dot(z, z)   # , /len(theta) drop constants for speed


def prior_tot(theta, mu, sigma, **kwargs):
    r""" 
    Wrapper for log_prior_gaussian() function. 
    Additional check for the +-3*sigma range. 
    If parameter alpha_cool, WD cooling delay, 

    Parameters:
    -----------
    - theta: list of floats
        Parameter values. 
    - mu: list of floats
        Means. 
    - sigma: list of floats
        Standard deviations. 

    Returns:
    --------
    Returns:
    --------
    - float
        Summed log Gaussian prior.  
    """
    lower = mu - 3*sigma

    # alpha_cool must be >= 0 !

    # get the location of alpha_cool index
    dcool_idx = kwargs.get('dcool_idx',None)

    if dcool_idx is not None and theta[dcool_idx] < 0: 
        # when it's negative, it will fall out of the range, because my mu=0
        # ugly way of implementing this...
        lower[dcool_idx] = mu[dcool_idx] 

    return log_prior_gaussian(theta, mu, sigma, lower=lower, upper=mu+3*sigma)


class HessProjLikelihood():
    """Class for constructing log likelihood, logL, based 
    the Gaia absolute CMD (Hess diagram) splitted into several populations (areas).

    logL consists of three terms: 
    - shape of a normalized color distribution (Hess x-projection)
    - shape of a normalized absolute magnitude distribution (Hess y-projection)
    - counts (number of stars of each CMD-population)
    """

    def __init__(self,sigma_shape2,epsilon_count,bin_width,ind_c=100,ind_m=117):
        r""" 
        Initialization

        Parameters:
        -----------
        - sigma_shape2: dict of dict of floats
            A systematic term to the dispersion of the logL_shape.
            Regulates how strictly the mismatch in color and magnitude shapes will be penalized. 
            Structure: sigma_shape2 = {'pop_name1':{'cdf':0.02,'mdf':0.01},'pop_name2':...}
        - epsilon_count: dict of floats
            Same but for logL_count. Can be viewed as an expected mismatch in star counts
            due to systematic error effects. 
        - bin_width: dict of floats
            Widths of the color and magnitude bins. 
            Structure: {'cdf':wbin_color,'mdf':wbin_mag}. 
        - ind_c: int
            Max index of the color distribution histogram where it should be cut off. 
            By default 100, this corresponds to G - G_RP = 1.55 given 
            my grid (-0.4,1.65) with a step of 0.02 mag. 
            Cut is applied because there is essentially no synthetic data for redder colors
            in Padova isochrones, so it doesn't make sense to compare data to zero model
            predictions. But I don't want to set the overall grid boundary to 1.55 as I want
            to show the noncut Hess diagram in the paper. 
        - ind_m: int
            Same for absolute magnitudes. By default no cut is applied, 
            index values is set to 117 - the last bin. 
        """
        self.ind_c = ind_c
        self.ind_m = ind_m
        self.ind_lim = {'cdf':self.ind_c,'mdf':self.ind_m}

        # Load precalculated histograms for the defined populations
        # in the modeled volume
        self.hessproj_data = self.read_cmdf_data(ind_lim=self.ind_lim)

        self.sigma_shape2 = sigma_shape2
        self.epsilon_count = epsilon_count
        self.bin_width = bin_width

        self.pops = list(self.hessproj_data.keys())

        # Precompute total sigmas for shape likelihoods (based on data and my parameter)
        self.sigma_shape_tot2 = {
            pop: {
                'cdf':self.hessproj_data[pop]['ncdf']/self.hessproj_data[pop]['cdf'] + self.sigma_shape2[pop]['cdf'],
                'mdf':self.hessproj_data[pop]['nmdf']/self.hessproj_data[pop]['mdf'] + self.sigma_shape2[pop]['mdf']
                } for pop in self.pops
        }
        
        # Precalculate total count of CMD population
        self.ntot_data = {pop: np.sum(self.hessproj_data[pop]['cdf']) for pop in self.pops} # same for cdf and mdf!


    def read_cmdf_data(
            self,
            pops=['wd','ms','ums','lms','g'],
            source_dir='./data/1kpc/',
            ind_lim={'cdf':100,'mdf':117}
            ):
        r""" 
        Load precalculated histograms for the defined populations in the modeled volume. 

        Parameters:
        -----------
        - pops: list of strings
            Names of populations. By default ['wd','ms','ums','lms','g']. 
            I think if some of them will be removed, the code will break,
            so don't change it. 
        - source_dir: string
            Relative path to the folder with the histograms. By default './data/1kpc/'. 
        - ind_lim: dict of ints
            Max indices of color and magnitude distributions where they should be cut off. 
            See HessProjLikelihood.__init__() for more details. 
            Structure: {'cdf':100,'mdf':117}, here default values are given. 

        Returns:
        --------
        - dict of dicts of 1d-arrays of floats
            Structure: 
            {'pop_name1':{
                'cdf':smoothed_color_distribution,
                'ncdf':smoothed_area_normalized_color_distribution,
                'mdf':smoothed_absmag_distribution,            
                'nmdf':smoothed_area_normalized_absmag_distribution,
                },
            'pop_name2':...
            }
        """
        coldata_d = {pop:{'SCD_S0':[],'SCD_S1':[],'NSCD_S0':[],'NSCD_S1':[]} for pop in pops}
        magdata_d = {pop:{'SMD_S0':[],'SMD_S1':[],'NSMD_S0':[],'NSMD_S1':[]} for pop in pops}

        for pop in pops:
            # Read color distribution
            cdf = np.loadtxt(source_dir + 'coldist/nsdf_' + pop + '.txt')
            coldata_d[pop]['SCD_S0'] = cdf[:,2] # Smoothed, full sample
            coldata_d[pop]['SCD_S1'] = cdf[:,3] # Smoothed, binary-clean subsample
            coldata_d[pop]['NSCD_S0'] = cdf[:,4] # Smoothed, full sample, area normalized
            coldata_d[pop]['NSCD_S1'] = cdf[:,5] # Smoothed, binary-clean subsample, area normalized

            # Read abs magnitude distribution
            mdf = np.loadtxt(source_dir + 'magdist/nsdf_' + pop + '.txt')
            magdata_d[pop]['SMD_S0'] = mdf[:,2]
            magdata_d[pop]['SMD_S1'] = mdf[:,3]
            magdata_d[pop]['NSMD_S0'] = mdf[:,4]
            magdata_d[pop]['NSMD_S1'] = mdf[:,5]

        coldata_d['LOW'] = np.round(cdf[:,0],2)
        coldata_d['HIGH'] = np.round(cdf[:,1],2)

        magdata_d['LOW'] = np.round(mdf[:,0],2)
        magdata_d['HIGH'] = np.round(mdf[:,1],2)

        eps = 1e-5 # small const to avoid zero division
        hessproj_data = {
            pop:{'cdf':coldata_d[pop]['SCD_S0'][:ind_lim['cdf']] + eps,
                 'ncdf':coldata_d[pop]['NSCD_S1'][:ind_lim['cdf']] + eps,
                 'mdf':magdata_d[pop]['SMD_S0'][:ind_lim['mdf']] + eps,
                 'nmdf':magdata_d[pop]['NSMD_S1'][:ind_lim['mdf']] + eps
                 } for pop in pops
            }

        return hessproj_data


    def lproj_shape(self,hessproj_model,pop,proj_type='cdf'):
        r""" 
        Calculate log likelihood term from the shape of the color 
        or absolute magnitude distribution of the CMD-defined population. 

        Parameters:
        -----------
        - hessproj_model: dict of dicts of 1d-arrays of floats
            Model prediction. Same structure as of the output of self.read_cmdf_data() method. 
        - pop: string
            Name of the CMD-defined population for which the logL is calculated. 
        - proj_type: string
            Type of the distribution to be analyzed: 'cdf' for color, 'mdf' for magnitude. 

        Returns:
        --------
        - float
            Log likelihood (arbitrary units). 
        - 1d-array of floats
            Contributions from each bin, useful for testing and debugging. 
        """
        name = 'n' + proj_type
        per_unit = (self.hessproj_data[pop][name] - hessproj_model[pop][name])**2 / self.sigma_shape_tot2[pop][proj_type] * self.bin_width[proj_type]
        l_shape = -0.5 * np.sum(per_unit[np.isfinite(per_unit)])

        return l_shape, per_unit
    

    def lproj_count(self,hessproj_model,pop):
        r""" 
        Calculate log likelihood term from the star counts of the CMD-defined population.  

        Parameters:
        -----------
        - hessproj_model: dict of dicts of 1d-arrays of floats
            Model prediction. Same structure as of the output of self.read_cmdf_data() method. 
        - pop: string
            Name of the CMD-defined population for which the logL is calculated. 

        Returns:
        --------
        - float
            Log likelihood (arbitrary units). 
        - tuple of floats
            Number of stars in the data, number of stars in the model
        """
        ntot_model = hessproj_model[pop]['n_pop']
        sigma_count_tot2 = ntot_model + (self.epsilon_count[pop] * ntot_model)**2
        l_count = -0.5 * (self.ntot_data[pop] - ntot_model)**2 / sigma_count_tot2

        return l_count, (self.ntot_data[pop], ntot_model)
    

    def lproj_pop(self,hessproj_model,pop):
        r""" 
        Calculate total log likelihood for a specified CMD-defined population. 
        Wrapper around self.lproj_shape() and self.lproj_count(). 

        Parameters:
        -----------
        Parameters:
        -----------
        - hessproj_model: dict of dicts of 1d-arrays of floats
            Model prediction. Same structure as of the output of self.read_cmdf_data() method. 
        - pop: string
            Name of the CMD-defined population for which the logL is calculated. 

        Returns:
        --------
        - float 
            Log likelihood. 
        """
        # Likelihood per population
        l_shape_cdf = self.lproj_shape(hessproj_model,pop,proj_type='cdf')
        l_shape_mdf = self.lproj_shape(hessproj_model,pop,proj_type='mdf')
        l_count = self.lproj_count(hessproj_model,pop)

        return l_shape_cdf[0] + l_shape_mdf[0] + l_count[0]
    

    def lproj_tot(self,hessproj_model):
        r""" 
        Calculate total log likelihood summed over all CMD-defined population. 
        Wrapper around self.lproj_pop(). 

        Parameters:
        -----------
        Parameters:
        -----------
        - hessproj_model: dict of dicts of 1d-arrays of floats
            Model prediction. Same structure as of the output of self.read_cmdf_data() method. 

        Returns:
        --------
        - float 
            Log likelihood. 
        """
        # Total likelihood over all populations
        l_tot = 0
        for pop in self.pops:
            l_pop = self.lproj_pop(hessproj_model,pop)
            l_tot += l_pop
            
        return l_tot
    

def likelihood(
        params,
        p,
        a,
        inp,
        pop_tabs_ref,           
        mag_range,
        mag_step,
        mag_smooth,
        sfr_handler,
        imf_handler,
        pop_handler,
        par_handler,
        constructor,
        l_handler,
        vz_mag=None,
        **kwargs
        ):
    r""" 
    Calculation of the log likelihood given an updated set of the model parameters. 

    Parameters:
    -----------
    - params: list of floats
        List of new parameters. 
    - p: namedtuple
        JJ-model parameter object from the model initialization. 
    - a: namedtuple
        JJ-model helpers array object from the model initialization.
    - inp: dict of arrays 
        JJ-model input array aobject from the model initialization.
    - pop_tabs_ref: dict of pandas.DataFrame objects or Astropy tables
        Reference stellar assemblies tables for thin/thick disk and halo:
        {'d':table1, 't':table2, 'sh':table3}. Reference means that these tables were
        calculated with the reference SFR, IMF, IFMR, 
        and WD DA fraction of 0.8 without Teff dependence. 
    - mag_range: list of lists of floats
        Range of xy-magnitudes in the Hess diagram. 
        Built as [color_min,color_max], [absmag_min, absmag_max]. 
    - mag_step: list of floats
        Step size in xy-magnitudes for the Hess diagram.
    - mag_smooth: list of floats
        Sigma of the Gaussian kernel for smoothing the Hess diagram, in xy-magnitudes.
    - sfr_handler: helpers.SFRHandler object
        Object for SFR manipulation. 
    - imf_handler: helpers.IMFHandler object
        Object for IMF manipulation. 
    - pop_handler: helpers.PopHandler object
        Object for working with stellar assemblies (populations). 
    - par_handler: helpers.ParHandler object
        Object for working with parameters.
    - constructor: helpers.HessConstructor object
        Object for modeling and working with Hess diagram. 
    - l_handler: HessProjLikelihood object
        Object to calculate likelihood from CMD-defined populations. 
    - vz_mag: dict of 2d-arrays
        Dictionary with arrays (n_pop, nz) giving the volume at each z 
        for each population of the input stellar assemblies tables. Structure is 
        {'d':(n_pop_d, nz),'t':(n_pop_t, nz),'sh':(n_pop_sh, nz)}. 
        Arrays must be 2d because modeled volume depends on the absolute magnitude
        (approach as in our SJ21 paper). Needed when the volume is not a simple sphere. 
    - kwargs: dict
        Keyword arguments. Same kwargs that are passed to posterior() function,
        should be also given here. See ParHandler.prepare_posterior_kwargs(). 
    
    Returns:
    --------
    - float 
        Log likelihood. 
    """
    #print('---Entered likelihood func---')

    # Fill the structured parameter dict with the new values
    param_struct = par_handler.fill_param_struct(params)
    pop_kwargs = kwargs

    # Update SFR
    if 'sfr' in param_struct.keys():
        inp, SFR_new = sfr_handler.update_sfr(**param_struct['sfr'])
        pop_kwargs['SFR_new'] = SFR_new
        #print('Updated SFR')

    # Update IMF
    if 'imf' in param_struct.keys():
        _, (_, IMF_new) = imf_handler.update_imf(**param_struct['imf'])
        pop_kwargs['IMF_new'] = IMF_new
        #print('Updated IMF')

    # Solve Poisson eq. and get new potential and scale heights
    out = local_run(p,a,inp,save=False,status_progress=False)
    inp_tabs = extract_model_tables(out,inp)
    #print('Solved PE')
    
    # Update number densities of the stellar assemblies
    pop_tabs, indt = pop_handler.update_pop_tabs(param_struct,pop_tabs_ref,**pop_kwargs)
    #print('Updated pop tabs')
        
    # Calculate Hess xy-projections
    proj = constructor.generate_hessproj(
        pop_tabs,
        indt['d'],
        inp_tabs,
        mag_range,
        mag_step,
        mag_smooth,
        vz_mag=vz_mag,
        volume='slice',
        idx_pop=kwargs['ind_pop'],
        ind_lim={'col':100,'mag':-1}
        )
    
    # Calculate likelihood
    prob = l_handler.lproj_tot(proj)
    #print('Calculated likelihood: ',prob)

    # Save output if needed
    logfile = kwargs.get("logfile",None)
    if logfile:
        with open(logfile,"a") as lf:
            lf.write("".join(['{:<10}'.format(round(par,5)) for par in params]) + f'{round(prob,5)}\n')
    #print('---Leaving likelihood func---')

    return prob



def posterior(
        params,
        param_mean,
        param_sigma,
        p,
        a,
        inp,
        pop_tabs_ref,
        mag_range, 
        mag_step, 
        mag_smooth,
        sfr_handler,
        imf_handler,
        pop_handler,
        par_handler,
        constructor,
        l_handler,
        vz_mag=None,
        **kwargs
        ):
    r""" 
    Calculation of posterior. 

    Parameters:
    -----------
    - params: list of floats
        List of new parameters. 
    - params_mean: list of floats
        Prior mean values. 
    - params_sigma: list of floats
        Prior standard deviations. 
    - p: namedtuple
        JJ-model parameter object from the model initialization. 
    - a: namedtuple
        JJ-model helpers array object from the model initialization.
    - inp: dict of arrays 
        JJ-model input array aobject from the model initialization.
    - pop_tabs_ref: dict of pandas.DataFrame objects or Astropy tables
        Reference stellar assemblies tables for thin/thick disk and halo:
        {'d':table1, 't':table2, 'sh':table3}. Reference means that these tables were
        calculated with the reference SFR, IMF, IFMR, 
        and WD DA fraction of 0.8 without Teff dependence. 
    - mag_range: list of lists of floats
        Range of xy-magnitudes in the Hess diagram. 
        Built as [color_min,color_max], [absmag_min, absmag_max]. 
    - mag_step: list of floats
        Step size in xy-magnitudes for the Hess diagram.
    - mag_smooth: list of floats
        Sigma of the Gaussian kernel for smoothing the Hess diagram, in xy-magnitudes.
    - sfr_handler: helpers.SFRHandler object
        Object for SFR manipulation. 
    - imf_handler: helpers.IMFHandler object
        Object for IMF manipulation. 
    - pop_handler: helpers.PopHandler object
        Object for working with stellar assemblies (populations). 
    - par_handler: helpers.ParHandler object
        Object for working with parameters.
    - constructor: helpers.HessConstructor object
        Object for modeling and working with Hess diagram. 
    - l_handler: HessProjLikelihood object
        Object to calculate likelihood from CMD-defined populations. 
    - vz_mag: dict of 2d-arrays
        Dictionary with arrays (n_pop, nz) giving the volume at each z 
        for each population of the input stellar assemblies tables. Structure is 
        {'d':(n_pop_d, nz),'t':(n_pop_t, nz),'sh':(n_pop_sh, nz)}. 
        Arrays must be 2d because modeled volume depends on the absolute magnitude
        (approach as in our SJ21 paper). Needed when the volume is not a simple sphere. 
    - kwargs: dict
        Keyword arguments. See ParHandler.prepare_posterior_kwargs(). 

    Returns:
    --------
    - float
        Posterior value (arbitrary units). 
    - float
        Prior value. 
    """
    #print('---Entered posterior func---')
    
    # Find position of the alpha_cool parameter to give it to the prior func
    prior_kwargs = {}
    if 'alpha_cool' in par_handler.flat_param_list:
        prior_kwargs['dcool_idx'] =\
            np.where(np.array(par_handler.flat_param_list)=='alpha_cool')[0][0]

    # Calculate prior and check whether parameters are in allowed intervals
    log_prior = prior_tot(params,param_mean,param_sigma,**prior_kwargs)
    if not np.isfinite(log_prior):
        #print('Inf prior:' + str(params))
        return -np.inf, (-np.inf,)
    
    # If parameters are reasonable, proceed to the likelihood calculation
    log_likelihood = likelihood(
        params,
        p,
        a,
        inp,
        pop_tabs_ref,
        mag_range, 
        mag_step, 
        mag_smooth,
        sfr_handler,
        imf_handler,
        pop_handler,
        par_handler,
        constructor,
        l_handler,
        vz_mag=vz_mag,
        **kwargs
        )

    return log_prior + log_likelihood, (log_prior,)


def initialize_params(mode,nwalkers,ndim,**kwargs):
    r""" 
    MCMC walkers initialization. 

    Parameters:
    -----------
    - mode: string
        Type of initialization - 'blob' or 'random'. 
    - nwalkers: int
        Number of walkers. 
    - ndim: int
        Number of MCMC parameters. 
    - kwargs: dict
        Optional keyword arguments. 
        See helpers.ParHandler.prepare_initialization_kwargs(). 

    Returns:
    --------
    - 2d-array of floats
        Initial positions of walkers. 
    """

    p_mean = np.array(kwargs['params_mean'])
    p_sigma = np.array(kwargs['params_sigma'])
    p_names = np.array(kwargs['labels'])

    dcool_idx = None
    if 'alpha_cool' in p_names:
        dcool_idx = np.where(p_names=='alpha_cool')[0][0]

    if mode=='blob':
        f_sig = kwargs.get('f_sig',1e-4)
        pos = p_mean + f_sig*p_sigma*np.random.randn(nwalkers,ndim)
        if dcool_idx is not None:
            pos[:,dcool_idx] = np.abs(pos[:,dcool_idx])

    if mode=='random':
        pos = p_mean + 3*p_sigma*(2*np.random.rand(nwalkers,ndim)-1)
        lower = p_mean - 3*p_sigma
        if dcool_idx is not None:
            lower[dcool_idx] = p_mean
        pos = np.clip(pos, lower, p_mean + 3*p_sigma)
        
    return pos
    

def mcmc_runner(sampler,pos,ndim,n_max,iter_step=100):
    r""" 
    Execute MCMC routine. 

    Parameters:
    -----------
    - sampler: emcee.EnsembleSampler object
        Initialized MCMC sampler. 
    - pos : 2d-array
        Initial positions of walkers, see optimizer.initialize_params(). 
    - ndim : int
        Number of MCMC parameters. 
    - n_max: int
        Max number of iterations. 
    - iter_step: int
        Iteration step at which the autocorrelation times are calculated. By default 100.

    Returns:
    --------
    - emcee.EnsembleSampler object
        Sampler after the MCMC run is finished
    - list of floats
        Autocorrelation time calculated during the run. 
    - emcee state object
        Final state of the sampler. 
    """
    
    # Initialize counters and variables
    count = 0
    autocorr = []
    tau = np.inf
    old_tau = np.inf
    converged = False
    final_state = None
    
    # ------------ MCMC loop --------------
    for state in sampler.sample(pos, iterations=n_max, progress=True, skip_initial_state_check=True):
        
        # Perform convergence check
        # Only after every i-th iteration AND enough steps to get a meaningful tau
        if sampler.iteration % iter_step == 0:

            # Update state
            final_state = state 
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tau = np.mean(sampler.get_autocorr_time(tol=0))
                    autocorr.append(tau)

                    # Check convergence
                    if sampler.iteration >= 50 * ndim:
                        converged = np.all(tau * 100 < sampler.iteration)
                        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)

                    if sampler.iteration != 0 and sampler.iteration % 1000 == 0:
                        print('Iteration '+str(sampler.iteration)+':\t',
                              'tau = '+str(int(round(tau,0))),
                              ' N/50 = '+str(int(round(sampler.iteration/50,0)))
                              )

                # If converged, stop checking but continue running
                if converged and count == 0:
                    print('Procedure converged, iteration =', sampler.iteration)
                    count += 1
                    break

            except Exception:
                autocorr.append(np.nan)

            old_tau = tau

    return sampler, autocorr, final_state

