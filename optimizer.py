
import numpy as np
import warnings
from jjmodel.mwdisk import local_run
from helpers import extract_model_tables


def chi2(hess_data,hess_model):

    chi2_hess = (hess_data - hess_model)**2
    return - np.mean(chi2_hess)


def log_prior_gaussian(theta, mu, sigma, lower=None, upper=None):
    # theta, mu, sigma: 1D arrays, same length
    if lower is not None and np.any(theta < lower): 
        return -np.inf
    if upper is not None and np.any(theta > upper): 
        return -np.inf
    z = (theta - mu) / sigma
    return -0.5 * np.dot(z, z) / len(theta)  # drop constants for speed


def prior(theta, mu, sigma, **kwargs):
    lower = mu - 3*sigma
    # alpha_cool must be >= 0 !
    dcool_idx = kwargs.get('dcool_idx',None)
    if dcool_idx is not None and theta[dcool_idx] < 0: 
        lower[dcool_idx] = mu[dcool_idx]
    return log_prior_gaussian(theta, mu, sigma, lower=lower, upper=mu+3*sigma)


def log_likelihood_counts(D, M, eps=0.5, eps_clip=0.5):  # choose eps ~ 0.5..1, not 1 arbitrarily
    # eps is a *modelled* floor; pick it and keep it fixed across θ
    Dp = np.clip(D + eps, eps_clip, None)
    Mp = np.clip(M + eps, eps_clip, None)
    r = np.log10(Dp) - np.log10(Mp)

    # delta-method variance on log10(X+eps), with Var[X]≈M for Poisson
    var_y = M / ((np.log(10)**2) * (Mp**2))
    var_y = np.clip(var_y, eps_clip, None)
    bins_good = np.sum(var_y > eps_clip)

    print(np.sum(r*r), np.sum(var_y), bins_good)
    # 
    return -0.5 * np.sum(r*r / var_y) / bins_good


def log_likelihood_poisson(D, M, eps_clip=1e-12):
    M = np.clip(M, eps_clip, None)           # avoid log(0)
    # Terms with D_i = 0 contribute just -M_i (in logL); the formula below handles it.
    with np.errstate(divide='ignore', invalid='ignore'):
        term = D * np.log(D / M) - (D - M)
        term = np.where(D > 0, term, -M)  # when D=0: D*log(D/M)=0, remaining is -M
    val = -term.mean()
    if not np.isfinite(val):
        return -np.inf
    return val                    # equals -0.5*C up to a constant


def loglike_masked_avg_logchi2(D, M, eps=0.5, min_count=0.1, var_floor=0.1):
    # keep bins with signal in either data or model
    mask = (D >= min_count) | (M >= min_count)
    if not np.any(mask):
        return -np.inf   # or a benign value

    Dp = D[mask] + eps
    Mp = np.clip(M[mask] + eps, 1e-12, None)

    r = np.log10(Dp) - np.log10(Mp)

    var_y = M[mask] / ((np.log(10)**2) * (Mp**2))
    var_y = np.maximum(var_y, var_floor)

    per_bin = r*r / var_y

    # average per informative bin
    logL = -0.5 * np.mean(per_bin)
    return logL


def likelihood(params,p,a,inp,pop_tabs_ref,
               hess_ref,mag_range, mag_step, mag_smooth,
               sfr_handler,imf_handler,pop_handler,par_handler,constructor,**kwargs
               ):
    #print('---Entered likelihood func---')

    param_struct = par_handler.fill_param_struct(params)
    #print(param_struct)

    pop_kwargs = kwargs

    if 'sfr' in param_struct.keys():
        inp, SFR_new = sfr_handler.update_sfr(**param_struct['sfr'])
        pop_kwargs['SFR_new'] = SFR_new
        #print('Updated SFR')

    if 'imf' in param_struct.keys():
        _, (_, IMF_new) = imf_handler.update_imf(**param_struct['imf'])
        pop_kwargs['IMF_new'] = IMF_new
        #print('Updated IMF')

    out = local_run(p,a,inp,save=False,status_progress=False)
    inp_tabs = extract_model_tables(out,inp)
    #print('Solved PE')

    pop_tabs, indt = pop_handler.update_pop_tabs(param_struct,pop_tabs_ref,**pop_kwargs)
    #print('Updated pop tabs')

    hess = constructor.generate_hess(pop_tabs,indt['d'],inp_tabs,
                                     mag_range,mag_step,mag_smooth)
    #print('Modeled Hess')
    #prob = chi2(hess_ref,np.log10(hess+1))
    #prob = log_likelihood_counts(hess_ref, hess, eps=0.5, eps_clip=1e-12)
    #prob = log_likelihood_poisson(hess_ref, hess, eps_clip=1e-12)
    prob = loglike_masked_avg_logchi2(hess_ref, hess)
    #print('Calculated likelihood: ',prob)

    logfile = kwargs.get("logfile",None)
    if logfile:
        with open(logfile,"a") as lf:
            lf.write("".join(['{:<10}'.format(round(par,5)) for par in params]) + f'{round(prob,5)}\n')
    #print('---Leaving likelihood func---')
    return prob


def posterior(params,param_mean,param_sigma,p,a,inp,pop_tabs_ref,hess_ref,
              mag_range, mag_step, mag_smooth,
              sfr_handler,imf_handler,pop_handler,par_handler,constructor,**kwargs
             ):
    #print('---Entered posterior func---')
    #print('params: ',params)
    prior_kwargs = {}
    if 'alpha_cool' in par_handler.flat_param_list:
        prior_kwargs['dcool_idx'] =\
            np.where(np.array(par_handler.flat_param_list)=='alpha_cool')[0][0]

    # First calculate prior and check whether parameters were in the allowed interval
    log_prior = prior(params,param_mean,param_sigma,**prior_kwargs)
    if not np.isfinite(log_prior):
        #print('Inf prior:' + str(params))
        return -np.inf, (-np.inf,)
    
    # If parameters are reasonable proceed to likelihood calculation
    try:
        log_likelihood = likelihood(params,p,a,inp,pop_tabs_ref,
                hess_ref,mag_range, mag_step, mag_smooth,
                sfr_handler,imf_handler,pop_handler,par_handler,constructor,**kwargs)
    except:
        return -np.inf, (-np.inf,)

    return log_prior + log_likelihood, (log_prior,)


def initialize_params(mode,nwalkers,ndim,**kwargs):

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
    

def mcmc_runner(sampler,pos,ndim,n_max):

    converged = False
    tau = np.inf
    old_tau = np.inf
    autocorr = []
    count = 0

    for _ in sampler.sample(pos, iterations=n_max, progress=True, skip_initial_state_check=True):
        # Only check every 100 iterations AND after enough steps to get a meaningful tau
        if sampler.iteration % 100 == 0:
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

                if converged and count == 0:
                    print('Procedure converged, iteration =', sampler.iteration)
                    count += 1
            except Exception:
                autocorr.append(np.nan)

            old_tau = tau

    return sampler, autocorr

