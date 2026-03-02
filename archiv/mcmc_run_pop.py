# %%
import os
import emcee
import warnings
import numpy as np 
import matplotlib.pyplot as plt
from multiprocessing import Pool

from jjmodel.tools import Timer
from jjmodel.input_ import p, a, inp
from jjmodel.mwdisk import disk_builder, local_run
from optimizer import HessProjLikelihood, posterior, initialize_params, mcmc_runner, likelihood
from helpers import (ParHandler, IFMRHandler, MSAgeHandler, extract_model_tables,
                     IMFHandler, SFRHandler, PopHandler, MCMCLogger, HessConstructor)
from prior import prior

# %%
timer = Timer()
t1 = timer.start()

# %%
# Parameters
# ---------------------------------------------------------------------
# General
mode_iso = 'Padova'  # Isochrones for MS and giants; WD always Montreal
mode_pop = 'pops_joined'  # Modeled populations: 'tot' = all, 'wd' = only WDs, 'ms' = only MS + giants, 
                 # 'ms+wd_joined' = both with joined likelihood
FeH_scatter = 0.07   # Scatter added to AMR of thin and thick disks
Nmet_dt = 7          # Number of metallicities per age bin
#radius = 50          # Radius of the modeled sphere, pc
mag_range = [[-0.5,1.65],[-5,18.5]]   # Hess diagram xy-ranges in (G-G_RP, M_G), mag
mag_step = [0.02,0.2]               # Steps in (G-G_RP, M_G), mag
mag_smooth = [0.06,0.8]             # Smoothing window size in (G-G_RP, M_G), mag
# WD 
f_da_teff = False    # If True fraction of DA/DB WDs is a function of Teff
age_ms_param_file = 'MS_lifetime_padova_new_metgrid/analysis/'+\
                    'fit_v1_Mbr1.18/tau_ms_params_v1_Mbr1.18.txt' # Parameters for MS lifetime fits
# MCMC setup
mode_init = 'blob'          # Blob arournd means or random
blob_f_sig = 0.01            # Defines blob size
n_max = 5000                  # Max number of iterations
dir_out = 'output/mcmc_pop2'     # Dir for output
save_log = False             # Save all tested parameter combinations
# ---------------------------------------------------------------------

# %%
# Choose parameters for MCMC optimization
# ------------------------------------------
par_optim = {
    'ifmr':         ['alpha1','alpha2','alpha3'], # 'm_br1', 'm_br2', 'alpha1' , 'alpha2', 'alpha3'
    'dcool':        ['alpha_cool'],
    'f_dadb':       ['f_da'],
    'sfr': {'d':    ['dzeta','eta','td2','sigmad','sigmap0'], # ,'sigmap0','tpk0', 
#            't':    ['gamma','beta','tt1']
            },
    'imf':          ['a0', 'a1', 'a2'] # , 'm0', 'm1'  
}

par_handler = ParHandler(par_optim,prior)
labels = par_handler.get_flat_param_list()
params_mean, params_sigma = par_handler.get_prior_for_params()

# %%
# Finish MCMC setup based on parameter list
ndim = len(labels)
nwalkers = 4*ndim
#n_cores = 28
n_cores = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))

# Create output directory
os.makedirs(dir_out,exist_ok=True)

# Create logger
logger = MCMCLogger(dir_out=dir_out)
logfile = logger.manage_logfile(save_log)

# Save simulation card 
logger.save_simulation_card(
    par_optim,mode_iso=mode_iso,mode_pop=mode_pop,
    FeH_scatter=FeH_scatter,Nmet_dt=Nmet_dt, # radius=radius,
    mag_range=mag_range,mag_step=mag_step,mag_smooth=mag_smooth,
    age_ms_param_file=age_ms_param_file,f_da_teff=f_da_teff,
    save_log=save_log,logfile=logfile,
    mode_init=mode_init,blob_f_sig=blob_f_sig,
    ndim=ndim,nwalkers=nwalkers,n_cores=n_cores,n_max=n_max
)

# %%
# Initialize SFR, IMF and population handlers

ifmr_handler = IFMRHandler()
msage_handler = MSAgeHandler(param_file=age_ms_param_file)
imf_handler = IMFHandler(p)
sfr_handler = SFRHandler(p, a, inp)
pop_handler = PopHandler(p, a, inp)
constructor = HessConstructor(p, a, mag_range, mag_step)


sigma_shape2 = {'wd':{'cdf':0.02,'mdf':0.01},
                'g':{'cdf':0.04,'mdf':0.02},
                'ums':{'cdf':0.02,'mdf':0.01},
                'ms':{'cdf':0.02,'mdf':0.01},
                'lms':{'cdf':0.04,'mdf':0.02},
                }
epsilon_count = {'wd':0.05,
                 'g':0.1,
                 'ums':0.05,
                 'ms':0.05,
                 'lms':0.1,
                }

bin_width = {'cdf':mag_step[0],'mdf':mag_step[1]}
l_handler = HessProjLikelihood(sigma_shape2,epsilon_count,bin_width)


SFR_ref = sfr_handler.create_reference_sfr()
imf_ref, (mass_binsc, IMF_ref) = imf_handler.create_reference_imf()

# Load observed Hess diagram 
#hess_ref = np.loadtxt('./data/hess/hess_tot.txt')
'''
if mode_pop == 'ms+wd_joined':
    hess_ref_wd = np.loadtxt('./data/hess/hess_wd.txt') # [:95,:95]
    hess_ref_ms = np.loadtxt('./data/hess/hess_ms.txt') # [:95,:95]
    hess_ref = {'wd':hess_ref_wd, 'ms':hess_ref_ms}
elif mode_pop == 'wd':
    hess_ref = np.loadtxt('./data/hess/hess_wd.txt')
elif mode_pop == 'ms':
    hess_ref = np.loadtxt('./data/hess/hess_ms.txt')
''';

# %%
# Calculate vertical disk structure
disk_builder(p,a,inp,status_progress=True)

# Prepare parameters for generating populations
pop_kwargs = par_handler.prepare_population_kwargs(
    FeH_scatter=FeH_scatter,
    Nmet_dt=Nmet_dt,
    mode_pop=mode_pop
    )

# Create population tables
pop_tabs_ref = pop_handler.create_reference_pop_tabs(imf_ref, mode_iso, **pop_kwargs) 

# Create reference copies for important columns
#pop_tabs_ref = pop_handler.create_reference_columns(pop_tabs_ref,['N', 'Mini', 'age'])
pop_tabs_ref = pop_handler.create_reference_columns(pop_tabs_ref,['N', 'Mini', 'age', 'age_WD'])

# Prepare idex columns for reference ages and initial masses
indt, indm = pop_handler.get_age_mass_idx(pop_tabs_ref,mass_binsc)

#print(pop_tabs_ref['d']['N'].sum())
#print(pop_tabs_ref['t']['N'].sum())
#print(pop_tabs_ref['sh']['N'].sum())

# Define DA/DB WD indices
ind_wd = pop_handler.make_wd_idx_dict(pop_tabs_ref)
if mode_pop == 'ms+wd_joined' or mode_pop == 'pops_joined':
    idx_wd, idx_ms = pop_handler.separate_wd_ms_idx(pop_tabs_ref,ind_wd)
else:
    idx_wd, idx_ms = None, None
pop_handler.display_wd_stats(mode_pop,ind_wd)

idx_pop = pop_handler.split_into_pops(pop_tabs_ref)
idx_pop_combined = pop_handler.combine_pops_indices(pop_tabs_ref,idx_pop)
#pop_id = pop_handler.create_pop_id_column(pop_tabs_ref,idx_pop_combined)

# Volume completeness data
d_mg_lim = np.loadtxt('./completeness/gdr3_dlim_vs_MG.txt').T
vz_grid = np.loadtxt('./completeness/vz_grid_myvolume.txt').T
vz_mag = pop_handler.get_vz_mag(pop_tabs_ref, d_mg_lim, vz_grid)


# %%
# Prepare parameters for posterior calculation
kwargs_post = par_handler.prepare_posterior_kwargs(
    SFR_ref,IMF_ref,indt,indm,
    save_log=save_log,logfile=logfile,
    mode_pop=mode_pop,ind_wd=ind_wd,
    ind_pop=idx_pop_combined, 
    #ind_mswd=(idx_ms,idx_wd),
    #pop_id=pop_id,
    f_da_teff=f_da_teff,
    ifmr_handler=ifmr_handler,
    msage_handler=msage_handler
)

# Define posterior function for MCMC
def probability_for_mcmc(theta):
    post, blob = posterior(
        theta,params_mean,params_sigma,p,a,inp,
        pop_tabs_ref, 
        mag_range, mag_step, mag_smooth,
        sfr_handler, imf_handler, pop_handler, par_handler, constructor,
        l_handler=l_handler, vz_mag=vz_mag, 
        #hess_ref=hess_ref
        **kwargs_post
    )
    if not np.isfinite(post):
        return -np.inf, blob
    return post, blob


# %%
# Initialize MCMC parameters
kwargs_init = par_handler.prepare_initialization_kwargs(
    mode_init,
    params_mean,
    params_sigma,
    labels,
    blob_f_sig=blob_f_sig
)
pos = initialize_params(mode_init, nwalkers, ndim, **kwargs_init)


# %%
# Run MCMC!
pool = Pool(processes=n_cores) 
sampler = emcee.EnsembleSampler(nwalkers, ndim, probability_for_mcmc, pool=pool)

# If resuming from saved state, load it
#pos = logger.load_state('output/mcmc')

sampler, autocorr, final_state = mcmc_runner(sampler,pos,ndim,n_max)

# Save final sampler state
logger.save_state(final_state)

# %%
# Check MCMC performance
# -------------------------

# Get MCMC chains, prior, and posterior 
chains_flat = logger.get_chains(sampler)

# Integrated autocorrelation time and acceptance fraction
logger.get_run_stats(sampler,autocorr)

# Best parameters
logger.get_best_params(chains_flat,labels,params_mean,params_sigma);

# %%
print('Output saved')
print(f'Execution time: {timer.stop(t1)}')

# %%



