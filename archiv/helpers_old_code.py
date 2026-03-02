import numpy as np
import warnings
from jjmodel.mwdisk import local_run
from helpers import extract_model_tables


def log_likelihood_counts(D, M, eps=0.5, eps_clip=0.5):  
    # choose eps ~ 0.5..1, not 1 arbitrarily
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


def softmin(logL1, logL2, T=0.5):
    # smaller T -> closer to hard min; larger T -> smoother average
    vals = np.array([-logL1 / T, -logL2 / T])
    a = np.max(vals)
    # stable log-sum-exp
    s = a + np.log(np.exp(vals - a).sum())
    return - T * s


def combined_loglike_lp(logL1, logL2, p=2.0):
    e1, e2 = -logL1, -logL2   # positive "badness"
    E_p = (e1**p + e2**p)**(1.0/p)
    return -E_p


def loglike_masked_avg_logchi2(D, M, eps=0.1, min_count=1e-2, var_floor=1e-2):
    # keep bins with signal in either data or model
    mask = (M >= min_count) # (D >= min_count) | (M >= min_count)
    if not np.any(mask):
        return -np.inf   # or a benign value

    Dp = D[mask] + eps
    Mp = M[mask] + eps

    w_scale = Dp * np.log10(D[mask] + 1)
    #w_scale = np.log10(D[mask] + 1)
    #w_scale = Dp**2 
    #w_scale = np.array([1])
    weights = w_scale / w_scale.sum()  # sums to 1

    r = np.log10(Dp) - np.log10(Mp)
    var_y = M[mask] / ((np.log(10)**2) * (Mp**2))
    var_y = np.maximum(var_y, var_floor)
    per_bin = r*r / var_y

    sigma_rel = 0.02
    diff_rel =  (Dp.sum() - Mp.sum()) / Dp.sum()
    logL_count = -0.5 * (diff_rel / sigma_rel)**2 #- 0.5 * np.log(2 * np.pi * sigma_rel**2)

    # average per informative bin
    logL_shape = -0.5 * np.sum(weights * per_bin) 

    fw1 = 1 # 1e-3
    fw2 = 1 # 1e-2 # larger contribution from logL_count! 
    scale = 0 # -5000
    logL = fw1*logL_shape + fw2*(logL_count + scale)

    print(round(M.sum(),1), '\t', 
          round(fw1*logL_shape,3), '\t', 
          round(fw2*(logL_count + scale),3), '\t', 
          round(logL,3)
          )

    return logL


def loglike_masked_combined(D, M, eps=0.1, min_count=1e-2):
    # keep bins with signal in either data or model
    #mask = (D >= min_count) | (M >= min_count)
    mask = (M >= min_count)
    if not np.any(mask):
        return -np.inf   # or a benign value

    Dp = D[mask] + eps
    Mp = M[mask] + eps

    # Plottable version of Dp
    Dp_im = D.copy()
    Dp_im += eps
    #i1, i2 = np.where((M <= min_count)) 
    #Dp_im[i1,i2] = np.nan

    # Plottable version of Mp
    Mp_im = M.copy()
    Mp_im += eps
    #Mp_im[i1,i2] = np.nan

    hx_d = np.nansum(Dp_im,axis=0)
    hy_d = np.nansum(Dp_im,axis=1)

    hx_m = np.nansum(Mp_im,axis=0)
    hy_m = np.nansum(Mp_im,axis=1)

    hx_dc = np.cumsum(hx_d)/hx_d.sum()
    hy_dc = np.cumsum(hy_d)/hy_d.sum()

    hx_mc = np.cumsum(hx_m)/hx_m.sum()
    hy_mc = np.cumsum(hy_m)/hy_m.sum()

    logl_xc = (hx_dc - hx_mc)**2
    logl_yc = (hy_dc - hy_mc)**2

    logL_shape = -0.5 * sum(logl_xc) -0.5 * sum(logl_yc)

    sigma_rel = 0.02
    diff_rel =  (D.sum() - M[mask].sum()) / D.sum()
    logL_count = -0.5 * (diff_rel / sigma_rel)**2

    fw1 = 1 # 1e-3
    fw2 = 5e-2 # 1e-2 # larger contribution from logL_count! 
    scale = 0 # - 2 / fw2 # -5000
    logL = fw1*logL_shape + fw2*(logL_count + scale)

    print(round(M.sum(),1), '\t', 
          round(fw1*logL_shape,3), '\t', 
          round(fw2*(logL_count + scale),3), '\t', 
          round(logL,3)
          )

    return logL


def hess_xy_proj(h_d, h_m, var_floor=1e-2):

    w = np.log10(h_d + 1)
    w /= np.nansum(w)

    r2 = (h_d - h_m)**2
    var = np.maximum(h_m, var_floor)
    pb = r2 / var
    logl = w*pb

    return logl



def loglike_masked_proj(D, M, eps=0.1, var_floor=1e-2, min_count=1e-2, fw=1, idx_lim=(None,None)):

    # Only work with bins with sufficient counts in both data and model
    i1, i2 = np.where((M <= min_count) & (D <= min_count)) # (D <= min_count) & 

    # Plottable version of Dp
    Dp_im = D.copy()
    Dp_im += eps
    Dp_im[i1,i2] = np.nan

    # Plottable version of Mp
    Mp_im = M.copy()
    Mp_im += eps
    Mp_im[i1,i2] = np.nan

    # Color distributions
    hx_d = np.nansum(Dp_im,axis=0)
    hx_m = np.nansum(Mp_im,axis=0)

    # Magnitude distributions
    hy_d = np.nansum(Dp_im,axis=1)
    hy_m = np.nansum(Mp_im,axis=1)

    logl_x = hess_xy_proj(hx_d, hx_m, var_floor=var_floor)
    logl_y = hess_xy_proj(hy_d, hy_m, var_floor=var_floor)

    if idx_lim[0] is None and idx_lim[1] is None:
        idx_lim = (len(logl_x),len(logl_y))
    
    logL = -0.5 * (np.nansum(logl_x[:idx_lim[0]]) + np.nansum(logl_y[:idx_lim[1]])) #  [:95] [:82]

    logL = fw * logL

    #print(round(M.sum(),1), '\t', round(logL,3), '\t')
    
    return logL


def likelihood(
        params,
        p,
        a,
        inp,
        pop_tabs_ref,           
        hess_ref,
        mag_range,
        mag_step,
        mag_smooth,
        sfr_handler,
        imf_handler,
        pop_handler,
        par_handler,
        constructor,
        l_handler=None,
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

    if kwargs['mode_pop'] == 'ms+wd_joined':

        ind_ms, ind_wd = kwargs['ind_mswd'] 
        
        pop_tabs_wd = {l:pop_tabs[l][ind_wd[l]] for l in pop_tabs.keys()}
        pop_tabs_ms = {l:pop_tabs[l][ind_ms[l]] for l in pop_tabs.keys()}

        hess_wd = constructor.generate_hess(pop_tabs_wd,indt['d'][ind_wd['d']],inp_tabs,
                                            mag_range,mag_step,mag_smooth)
        #prob_wd = loglike_masked_avg_logchi2(hess_ref['wd'], hess_wd)
        #prob_wd = loglike_masked_combined(hess_ref['wd'], hess_wd)
        prob_wd = loglike_masked_proj(hess_ref['wd'], hess_wd)

        hess_ms = constructor.generate_hess(pop_tabs_ms,indt['d'][ind_ms['d']],inp_tabs,
                                            mag_range,mag_step,mag_smooth)
        #prob_ms = loglike_masked_avg_logchi2(hess_ref['ms'], hess_ms)
        #prob_ms = loglike_masked_combined(hess_ref['ms'], hess_ms)
        prob_ms = loglike_masked_proj(hess_ref['ms'], hess_ms, fw=5e-2, idx_lim=(95,82))

        prob = prob_wd + prob_ms
        #prob = softmin(prob_wd,prob_ms,T=1.0)
        #prob = combined_loglike_lp(prob_wd, prob_ms, p=2.0)

        #ksi = 1/3 
        #prob = ksi*prob_ms + (1 - ksi)*prob_wd

    elif kwargs['mode_pop'] == 'pops_joined':
        
        # calculate total Hess
        
        '''
        hess_tot = constructor.generate_hess(
            pop_tabs,indt['d'],inp_tabs,
            mag_range,mag_step,mag_smooth,
            volume='slice',
            vz_mag=vz_mag
            )
        
        # color-magnitude distributions
        proj = constructor.proj_from_hess(
            hess_tot,
            range={'col':[-0.5,1.5],'mag':[-5,18.0]},
            smooth={'col':mag_smooth[0],'mag':mag_smooth[1]},
            ind_lim={'col':100,'mag':-1}
            )
        '''
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
        
        prob = l_handler.lproj_tot(proj) # np.random.rand() #
        #print('N_WD:', int(hessproj_model['wd']['cdf'].sum()))

    else:
   
        hess = constructor.generate_hess(pop_tabs,indt['d'],inp_tabs,
                                         mag_range,mag_step,mag_smooth)
        #prob = log_likelihood_counts(hess_ref, hess, eps=0.5, eps_clip=1e-12)
        #prob = log_likelihood_poisson(hess_ref, hess, eps_clip=1e-12)
        #prob = loglike_masked_avg_logchi2(hess_ref, hess)
        #prob = loglike_masked_combined(hess_ref, hess)
        if kwargs['mode_pop'] == 'ms':
            prob = loglike_masked_proj(hess_ref, hess, fw=1e-1, idx_lim=(95,82))
        else:
            prob = loglike_masked_proj(hess_ref, hess)
    
    #print('Calculated likelihood: ',prob)

    logfile = kwargs.get("logfile",None)
    if logfile:
        with open(logfile,"a") as lf:
            lf.write("".join(['{:<10}'.format(round(par,5)) for par in params]) + f'{round(prob,5)}\n')
    #print('---Leaving likelihood func---')
    return prob



#################


l = []
pr = []

s = np.round(np.arange(-2.0,2.0,0.1),1)

for i in range(len(s)):

    p_test = params_mean + s[i]*params_sigma
    if s[i] < 0: 
        p_test[3] = 0
    p1, p2 = probability_for_mcmc(p_test)  # Test run
    try:
        l.append(p1 - p2)
        pr.append(p2)
        print(s[i], np.round(p1,2), np.round(l[-1],2)[0], np.round(p2,2)[0])
    except:
        print('Error at s =', s[i])
        pass
    print()

l_min = np.min(l)
l_max = np.max(l)

plt.figure()
plt.plot(s,l)
plt.plot(s,pr)
plt.plot(s,np.add(l,pr),c='k')
plt.ylim(-100,0)