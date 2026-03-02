# %%
import os
import emcee
import numpy as np 
from multiprocessing import Pool

from jjmodel.tools import Timer
from jjmodel.input_ import p, a, inp
from jjmodel.mwdisk import disk_builder

from prior import prior
from optimizer import HessProjLikelihood, posterior, initialize_params, mcmc_runner
from helpers import (ParHandler, IFMRHandler, MSAgeHandler, IMFHandler, 
                     SFRHandler, PopHandler, MCMCLogger, HessConstructor)

timer = Timer()
t1 = timer.start()

# %% [markdown]
# ### Spesify parameters of the run

# %%
# Parameters
# ---------------------------------------------------------------------
# General
mode_iso = 'Padova'       # Isochrones for MS and giants; WD always Montreal
mode_pop = 'pops_joined'  # Modeled populations: 'tot' = all, 'wd' = only WDs, 'ms' = only MS + giants, 
FeH_scatter = 0.07   # Scatter added to AMR of thin and thick disks
Nmet_dt = 7          # Number of metallicities per age bin
mag_range = [[-0.5,1.65],[-5,18.5]]   # Hess diagram xy-ranges in (G-G_RP, M_G), mag
mag_step = [0.02,0.2]                 # Steps in (G-G_RP, M_G), mag
mag_smooth = [0.06,0.8]               # Smoothing window size in (G-G_RP, M_G), mag

# WD 
f_da_teff = False    # If True fraction of DA/DB WDs is a function of Teff
age_ms_param_file = 'MS_lifetime_padova_new_metgrid/analysis/'+\
                    'fit_v1_Mbr1.18/tau_ms_params_v1_Mbr1.18.txt' # Parameters for MS lifetime fits
# MCMC setup
mode_init = 'blob'          # Blob arournd means or random
blob_f_sig = 0.01           # Defines blob size
n_max = 10                  # Max number of iterations
dir_out = 'output/mcmc'     # Dir for output
save_log = False            # Save all tested parameter combinations
n_cpu_local = 8             # For the local run when "SLURM_CPUS_ON_NODE" is not set
n_cores = int(os.environ.get("SLURM_CPUS_ON_NODE", n_cpu_local)) 

# Likelihood hyperparameters
sigma_shape2 = {
    'wd':{'cdf':0.02,'mdf':0.01},
    'g':{'cdf':0.02,'mdf':0.01},
    'ums':{'cdf':0.02,'mdf':0.01},
    'ms':{'cdf':0.02,'mdf':0.01},
    'lms':{'cdf':0.02,'mdf':0.01},
    }
epsilon_count = {
    'wd':0.05,
    'g':0.05,
    'ums':0.05,
    'ms':0.05,
    'lms':0.05,
}

# %% [markdown]
# ### Create structured dictionary with the model parameters to optimize
# 
# Full example: 
# ```
# par_optim = {
#     'ifmr':         ['alpha1','alpha2','alpha3','m_br1', 'm_br2'],
#     'dcool':        ['alpha_cool'],
#     'f_dadb':       ['f_da'],
#     'sfr': {'d':    ['dzeta','eta','td2','sigmad','sigmap0','tpk0','sigmap1','tpk1'], 
#             't':    ['gamma','beta','tt1']},
#     'imf':          ['a0', 'a1', 'a2','m0', 'm1']
#     }
# ```
# 
# Results of my tests:
# - Thick-disk SFR should not be optimized because I use a near-plane sample.
# - IFMR breakpoints ```m_br1``` and ```m_br2``` don't change, return prior values. So can be also skipped. 
# - Slopes ```alpha2``` and ```alpha3``` also remain essentially the same. 
# - ```f_da``` changes by a few percent. 
# - Thin-disk SFR parameters ```eta```, ```dzeta```, ```td2``` are correlated, so only two of them can be used. 
# - Smaller secondary peak parameters ```sigmap1``` and ```tpk1``` cannot be constrained, so no need to add them. 
# - IMF break points ```m0``` and ```m1``` also remain almost the same and can be skipped. 
# 

# %%
# Choose parameters for the MCMC optimization
par_optim = {
    'ifmr':         ['alpha1','alpha2','alpha3'],
    'dcool':        ['alpha_cool'],
    'f_dadb':       ['f_da'],
    'sfr': {'d':    ['dzeta','eta','sigmad','sigmap0','tpk0']},
    'imf':          ['a0', 'a1', 'a2']
}

par_handler = ParHandler(par_optim,prior)

# Flatten par_optim into a list
labels = par_handler.get_flat_param_list()
# Get the parameters' prior mean and standard deviation values
params_mean, params_sigma = par_handler.get_prior_for_params()


# %% [markdown]
# ### Do some more preparations

# %%
# Finish MCMC setup based on the parameter list
ndim = len(labels)
nwalkers = 4*ndim

# Create output directory
os.makedirs(dir_out,exist_ok=True)

# Create logger
logger = MCMCLogger(dir_out=dir_out)
logfile = logger.manage_logfile(save_log)

# Save simulation card 
# Helps to understand later what kind of simulation it was
logger.save_simulation_card(
    par_optim,
    mode_iso=mode_iso,
    mode_pop=mode_pop,
    FeH_scatter=FeH_scatter,
    Nmet_dt=Nmet_dt,
    mag_range=mag_range,
    mag_step=mag_step,
    mag_smooth=mag_smooth,
    age_ms_param_file=age_ms_param_file,
    f_da_teff=f_da_teff,
    save_log=save_log,
    logfile=logfile,
    mode_init=mode_init,
    blob_f_sig=blob_f_sig,
    ndim=ndim,
    nwalkers=nwalkers,
    n_cores=n_cores,
    n_max=n_max
)

# %%
# Initialize SFR, IMF and population, Hess, and likelihood handlers
# ------------------------------------------------------------------

imf_handler = IMFHandler(p)
ifmr_handler = IFMRHandler(a)
sfr_handler = SFRHandler(p, a, inp)
pop_handler = PopHandler(p, a, inp)
constructor = HessConstructor(p, a, mag_range, mag_step)
msage_handler = MSAgeHandler(param_file=age_ms_param_file)

bin_width = {'cdf':mag_step[0],'mdf':mag_step[1]}
l_handler = HessProjLikelihood(sigma_shape2,epsilon_count,bin_width)

# Define reference SFR and IMF (default parameters from SJ21)
SFR_ref = sfr_handler.create_reference_sfr()
imf_ref, (mass_binsc, IMF_ref) = imf_handler.create_reference_imf()

# %% [markdown]
# ### Run the local model and create stellar assemblies

# %%
# Calculate vertical disk structure for the defaul parameters
disk_builder(p,a,inp,status_progress=True)

# Prepare kwargs for generating stellar assemblies
pop_kwargs = par_handler.prepare_population_kwargs(
    FeH_scatter=FeH_scatter,
    Nmet_dt=Nmet_dt,
    mode_pop=mode_pop
    )

# Create stellar assemblies (reference) tables 
# (predictions for the reference SFR, IMF, Cummings+18 IFMR, 
# and DA fraction 0.8 without Teff dependence)
pop_tabs_ref = pop_handler.create_reference_pop_tabs(imf_ref, mode_iso, **pop_kwargs) 

# Create reference copies of the important columns
# These copies will be not modified during the run
pop_tabs_ref = pop_handler.create_reference_columns(pop_tabs_ref,['N', 'Mini', 'age', 'age_WD'])

# Prepare position index columns for the reference ages and initial masses
indt, indm = pop_handler.get_age_mass_idx(pop_tabs_ref,mass_binsc)

# Find DA/DB WD position indices
# idx_wd refers to WD, idx_ms - to all other stars
ind_wd = pop_handler.make_wd_idx_dict(pop_tabs_ref)
idx_wd, idx_ms = pop_handler.separate_wd_ms_idx(pop_tabs_ref,ind_wd)
pop_handler.display_wd_stats(mode_pop,ind_wd)

# Find position indices of CMD-defined populations:
# white dwarfs, giants, main sequence (MS), upper MS, and lower MS
idx_pop = pop_handler.split_into_pops(pop_tabs_ref)
idx_pop_combined = pop_handler.combine_pops_indices(pop_tabs_ref,idx_pop)

# Load volume (completeness) data
# Calculate vz_mag grid - for each row in pop_tabs_ref it gives volume accupied 
# by these type of stars at each z (corresponds to the modeled vertical grid)
d_mg_lim = np.loadtxt('./completeness/gdr3_dlim_vs_MG.txt').T
vz_grid = np.loadtxt('./completeness/vz_grid_myvolume.txt').T
vz_mag = pop_handler.get_vz_mag(pop_tabs_ref, d_mg_lim, vz_grid)


# %% [markdown]
# ### Define a posterior function for the MCMC

# %%
# Prepare parameters for posterior calculation
kwargs_post = par_handler.prepare_posterior_kwargs(
    SFR_ref,
    IMF_ref,
    indt,
    indm,
    save_log=save_log,
    logfile=logfile,
    mode_pop=mode_pop,
    ind_wd=ind_wd,
    ind_pop=idx_pop_combined, 
    f_da_teff=f_da_teff,
    ifmr_handler=ifmr_handler,
    msage_handler=msage_handler
)

# Define posterior
def probability_for_mcmc(theta):
    post, blob = posterior(
        theta,
        params_mean,
        params_sigma,
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
        **kwargs_post
    )
    if not np.isfinite(post):
        return -np.inf, blob
    return post, blob


# %% [markdown]
# ### Run the MCMC

# %%

# Create kwargs for the walkers initialization
kwargs_init = par_handler.prepare_initialization_kwargs(
    mode_init,
    params_mean,
    params_sigma,
    labels,
    blob_f_sig=blob_f_sig
)
# Initialize walkers
pos = initialize_params(mode_init, nwalkers, ndim, **kwargs_init)

# If resuming from a saved state, load it
#pos = logger.load_state('output/mcmc')

# Run MCMC
pool = Pool(processes=n_cores) 
sampler = emcee.EnsembleSampler(nwalkers, ndim, probability_for_mcmc, pool=pool)
sampler, autocorr, final_state = mcmc_runner(sampler,pos,ndim,n_max)

# Save final sampler state
logger.save_state(final_state)


# %% [markdown]
# ### Do a performance check and save the output

# %%

# Get MCMC chains, prior, and posterior 
chains_flat = logger.get_chains(sampler)

# Integrated autocorrelation time and acceptance fraction
logger.get_run_stats(sampler,autocorr)

# Best parameters
logger.get_best_params(chains_flat,labels,params_mean,params_sigma);

print('\nOutput saved')
print(f'Execution time: {timer.stop(t1)}')

# %%



