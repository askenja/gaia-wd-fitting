
import os
import warnings
import numpy as np 
from scipy import optimize
from scipy.stats import norm
from astropy.table import Table
from fast_histogram import histogram2d
from scipy.interpolate import RegularGridInterpolator

import matplotlib as mpl
import matplotlib.pyplot as plt

from jjmodel.constants import tp
from jjmodel.constants import KM, tr
from jjmodel.geometry import Volume
from jjmodel.funcs import IMF, SFR
from jjmodel.analysis import _extend_mag_
from jjmodel.tools import convolve2d_gauss
from jjmodel.populations import stellar_assemblies_r


class MCMCLogger:

    def __init__(self,dir_out='./mcmc_output'):

        self.dir_out = dir_out
        

    def manage_logfile(self,save_log):

        logfile = None
        if save_log:
            logfile = os.path.join(self.dir_out,'logfile.txt')
            with open(logfile,'w') as f:
                f.write('# Tested parameters\n')
        return logfile
    
    def save_simulation_card(self,
                             par_optim,
                             mode_iso='Padova',
                             mode_pop='tot',
                             radius=50,
                             FeH_scatter=0,
                             Nmet_dt=0,
                             mag_range=[[],[]],
                             mag_step=[],
                             mag_smooth=[],
                             age_ms_param_file='',
                             f_da_teff=False,
                             save_log=False,
                             logfile=None,
                             mode_init='blob',
                             blob_f_sig=None,
                             ndim=None,
                             nwalkers=None,
                             n_cores=None,
                             n_max=None):

        # Save simulation card 
        with open(os.path.join(self.dir_out,'simulation_card.txt'),"w") as f:
            f.write('{:<25}'.format('Main isochrones')+mode_iso+'\n')
            f.write('{:<25}'.format('Radius_sphere [pc]')+str(radius)+'\n')
            f.write('{:<25}'.format('WDs included')+str((mode_pop in ('wd','tot')))+'\n')
            f.write('{:<25}'.format('MS included')+str((mode_pop in ('wd+ms','tot')))+'\n')
            f.write('{:<25}'.format('FeH_scatter [dex]')+str(FeH_scatter)+'\n')
            if FeH_scatter != 0:
                f.write('{:<25}'.format('Nmet_dt+sh')+str(Nmet_dt)+'\n')
            f.write('{:<25}'.format('(G-RP,MG)_lim, [mag]')+str(mag_range)+'\n')
            f.write('{:<25}'.format('(G-RP,MG)_step, [mag]')+str(mag_step)+'\n')
            f.write('{:<25}'.format('(G-RP,MG)_sm, [mag]')+str(mag_smooth)+'\n')
            f.write('{:<25}'.format('Test[IFMR]')+str(('ifmr' in par_optim))+'\n')
            if 'ifmr' in par_optim:
                f.write('{:<25}'.format('Age_MS params')+age_ms_param_file+'\n')
            f.write('{:<25}'.format('Test[d_cool]')+str(('dcool' in par_optim))+'\n')
            f.write('{:<25}'.format('Test[f_DA/DB]')+str(('f_dadb' in par_optim))+'\n')
            f.write('{:<25}'.format('f_DA(Teff)')+str(f_da_teff)+'\n')
            f.write('{:<25}'.format('Test[SFR]')+str(('sfr' in par_optim))+'\n')
            f.write('{:<25}'.format('Test[IMF]')+str(('imf' in par_optim))+'\n')
            f.write('{:<25}'.format('Parameters')+'/best_parameters.txt'+'\n')
            f.write('{:<25}'.format('Parameter log')+(logfile if save_log else str(save_log))+'\n')
            f.write('{:<25}'.format('Initialization')+mode_init+'\n')
            if mode_init == 'blob' and blob_f_sig:
                f.write('{:<25}'.format('Blob_size')+str(blob_f_sig)+'\n')
            f.write('{:<25}'.format('N_parameters')+str(ndim)+'\n')
            f.write('{:<25}'.format('N_walkers')+str(nwalkers)+'\n')
            f.write('{:<25}'.format('N_CPU')+str(n_cores)+'\n')
            f.write('{:<25}'.format('N_max_iteration')+str(n_max)+'\n')


    def get_run_stats(self,sampler,autocorr,verbose=True):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                af = sampler.acceptance_fraction
                afm = round(np.nanmean(af),2)
                if verbose:
                    print('Acceptance_fraction = ', afm)
            except:
                afm = np.nan
                print('Acceptance_fraction could not be estimated')

            try:
                tau = sampler.get_autocorr_time(tol=0)
                taum = int(round(np.nanmean(tau),0))
                if verbose:
                    print('Integrated autocorrelation time = ', taum, '\n')
            except:
                print('Integrated autocorrelation time could not be estimated\n')
                taum = np.nan

        np.savetxt(os.path.join(self.dir_out,'autocorr_time.txt'),\
                   np.stack((np.arange(len(autocorr))*100,
                             autocorr,
                             np.arange(len(autocorr))*2
                             ),
                             axis=-1),
                   header = 'taum = ' + str(taum) + ', afm = ' + str(afm) +\
                            '\niteration, autocorrelation time, N/50'
                )
    

    def get_chains(self,sampler):

        # Get MCMC chains, prior and posterior 

        log_prior = sampler.get_blobs()
        log_posterior = sampler.get_log_prob()
        chains = sampler.get_chain()
        chains_flat = sampler.get_chain(flat=True)

        np.save(os.path.join(self.dir_out,'log_prior'),log_prior)
        np.save(os.path.join(self.dir_out,'log_posterior'),log_posterior)
        np.save(os.path.join(self.dir_out,'chains'),chains)

        return chains_flat
    
    
    def get_best_params(self,chains_flat,labels,params_mean,params_sigma,verbose=True):

        # Find new best parameter values:
        best_params, er1,er2 = [],[],[]

        if verbose:
            print('{:>20}'.format('value'), 
                '{:>10}'.format('+error'), 
                '{:>10}'.format('-error'), 
                '{:>10}'.format('max_error, %')
                )

        ndim = len(labels)
        for i in range(ndim):
            mcmc = np.percentile(chains_flat[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            best_params.append(round(mcmc[1],3))
            er1.append(round(q[0],3))
            er2.append(round(q[1],3))
            if verbose:
                print('{:>10}'.format(labels[i]), 
                    '{:>10}'.format(round(mcmc[1],3)),
                    '{:>10}'.format(round(q[0],3)),
                    '{:>10}'.format(round(q[1],3)),
                    '{:>10}'.format(round(max(q[0],q[1])/mcmc[1]*100,1))
                    )

        columns = ['Parameter','Prior mean','Prior sigma','Best value','Err+','Err-']
        with open(os.path.join(self.dir_out,'best_params.txt'),"w") as f:
            f.write('#' + ''.join((["{:<15}".format(el) for el in columns])) + '\n')
            for i in range(ndim):
                f.write(''.join((["{:<15}".format(el) for el in 
                                [labels[i],params_mean[i],params_sigma[i],best_params[i],er1[i],er2[i]]])) + '\n')



class ParHandler:

    def __init__(self,param_names,prior):

        param_struct = {}
        for key in param_names.keys():
            if key != 'sfr':
                param_struct[key] = {name:np.nan for name in param_names[key]}
            else:
                param_struct[key] = {}
                for sfr_key in param_names[key].keys():
                    param_struct[key][sfr_key] = {name:np.nan for name in param_names[key][sfr_key]}

        self.param_struct = param_struct
        self.prior = prior 


    def get_flat_param_list(self):
        flat_param_list = []
        for par_class in self.param_struct.keys():
            #print(par_class)
            if par_class != 'sfr':
                flat_param_list.extend(list(self.param_struct[par_class].keys()))
            else:
                flat_param_list.extend(list(self.param_struct[par_class]['d'].keys()))
                flat_param_list.extend(list(self.param_struct[par_class]['t'].keys()))
        
        self.flat_param_list = flat_param_list
        return self.flat_param_list


    def get_prior_for_params(self):

        params_mean = np.array([self.prior[key]['m'] for key in self.flat_param_list])
        params_sigma = np.array([self.prior[key]['s'] for key in self.flat_param_list])

        return params_mean, params_sigma
    

    def fill_param_struct(self,theta):
        """Update hierarchical parameter dictionary with values from a flat list."""

        if len(theta) != len(self.flat_param_list):
            raise ValueError('Length of parameter list does not match the number of parameters in structured dict!')

        i = 0
        for group, subdict in self.param_struct.items():
            # Case 1: simple subdict (e.g. ifmr, dcool, f_dadb, imf)
            if all(not isinstance(v, dict) for v in subdict.values()):
                for key in subdict.keys():
                    self.param_struct[group][key] = theta[i]
                    i += 1
            # Case 2: nested subdict (e.g. sfr -> d, t)
            else:
                for subgroup, subsubdict in subdict.items():
                    for key in subsubdict.keys():
                        self.param_struct[group][subgroup][key] = theta[i]
                        i += 1
        return self.param_struct
    

    def prepare_posterior_kwargs(self,
                                 SFR_ref,
                                 IMF_ref,
                                 indt,
                                 indm,
                                 save_log=False,
                                 logfile=None,
                                 mode_pop='tot',
                                 ind_wd=None,
                                 f_da_teff=False,
                                 ifmr_handler=None,
                                 msage_handler=None
                                ):

        kwargs_post = {'SFR_ref':SFR_ref,'IMF_ref':IMF_ref,'indt':indt,'indm':indm}
        if save_log:
            kwargs_post['logfile'] = logfile
        if mode_pop in ('wd','tot'):
            kwargs_post['ind_wd'] = ind_wd
        if 'f_dadb' in self.param_struct.keys():
            kwargs_post['f_da_teff'] = f_da_teff
        if 'ifmr' in self.param_struct.keys():
            kwargs_post['ifmr_handler'] = ifmr_handler
            kwargs_post['msage_handler'] = msage_handler

        return kwargs_post
    

    def prepare_population_kwargs(self, FeH_scatter=0, Nmet_dt=1, mode_pop='tot'):

        pop_kwargs = {'FeH_scatter':FeH_scatter}

        if FeH_scatter != 0:
            pop_kwargs['Nmet_dt'] = Nmet_dt
            pop_kwargs['Nmet_sh'] = Nmet_dt

        if mode_pop == 'tot':
            pop_kwargs['wd'] = 'ms+wd'
        elif mode_pop == 'wd':
            pop_kwargs['wd'] = 'wd'
        
        return pop_kwargs
    

    def prepare_initialization_kwargs(self,mode_init,
                                      params_mean,params_sigma,labels,blob_f_sig=1e-2):

        init_kwargs = {'params_mean':params_mean,'params_sigma':params_sigma,'labels':labels}
        if mode_init == 'blob':
            try:
                init_kwargs['f_sig'] = blob_f_sig 
            except:
                pass
        return init_kwargs


class HessConstructor:

    def __init__(self, r_max, p, a):

        self.p = p 
        self. a = a

        zlim = [0,r_max + 1.5*self.p.zsun]
        indz1, indz2 = int(abs(zlim[0])//self.p.dz), int(abs(zlim[1]//self.p.dz))
        self.indz1, self.indz2 = np.sort([indz1,indz2])

        # Define spherical grid
        V = Volume(self.p,self.a)
        self.volume = V.local_sphere(0,r_max)[0][self.indz1:self.indz2]


    def pops_in_volume(self,pop_tabs,indt,inp_tabs):
        r"""
        Calculates the number of stars in a volume.  

        Parameters:
        -----------
        - pop_tabs: dict
            Tables of populations with their keys.
        - indt: 1d-array
            Age indices for thin-disk population table. 
        - inp_tabs: dict
            Model predictions for potential, AVR, scale heights...

        Returns:
        --------
        Population tables with Nz column. 
        """
        
        tabd, tabt, tabsh = pop_tabs['d'],pop_tabs['t'],pop_tabs['sh']
        
        Fi_sliced = inp_tabs['Fi'][self.indz1:self.indz2]
        volume = self.volume
        jd_array = self.a.jd_array
        
        sigt = self.p.sigt
        sigsh = self.p.sigsh
        AVR  = inp_tabs['AVR']
        Hd, Ht, Hsh = inp_tabs['Hd'], inp_tabs['Ht'], inp_tabs['Hsh']

        wt = 0.5 / Ht * np.sum(np.exp(-Fi_sliced / KM**2 / sigt**2)*volume)
        wsh = 0.5 / Hsh * np.sum(np.exp(-Fi_sliced / KM**2 / sigsh**2)*volume) 

        exp_AVR = np.exp(-Fi_sliced[:, None] / KM**2 / AVR[jd_array]**2)  # shape: (z, jd)
        
        if self.p.pkey==1:
            Fp = inp_tabs['Fp']
            Hdp = inp_tabs['Hdp']
            sigp = self.p.sigp
            fpr0 = 1 - np.sum(Fp,axis=0) 

            wd_total = (fpr0[jd_array] / (2 * Hd[jd_array])) * np.sum(exp_AVR * volume[:, None], axis=0)

            exp_sigp = np.exp(-Fi_sliced[:, None] / KM**2 / sigp**2)  # shape: (z, npeak)
            wd_total += np.sum((Fp[:, jd_array] / (2 * Hdp[:, None])) *
                                np.sum(exp_sigp[:, :, None] * volume[:, None, None], axis=0),
                                axis=0
                              )
        else:
            wd_total =  0.5 / Hd[jd_array] * np.sum(exp_AVR * volume[:, None], axis=0)
                
        tabd['Nz'] = np.array(tabd['N']*wd_total[indt])
        tabt['Nz'] = np.array(tabt['N']*wt)
        tabsh['Nz'] = np.array(tabsh['N']*wsh)

        return {'d':tabd, 't':tabt, 'sh':tabsh}
    
    @staticmethod
    def hess_from_data(x_column,y_column,mag_range,mag_step,mag_smooth=None,weights=None):

        if weights is None:
            weights = np.ones_like(x_column)

        xlen = int(round(abs((mag_range[0][0]-mag_range[0][1])/mag_step[0]),0)) 
        ylen = int(round(abs((mag_range[1][0]-mag_range[1][1])/mag_step[1]),0))
        
        hess = np.zeros((xlen,ylen))
        hess = histogram2d(x_column,y_column,weights=weights,
                           bins=[xlen,ylen],range=mag_range)

        if mag_smooth is not None:                                        
            hess = convolve2d_gauss(hess,mag_smooth,mag_range)

        return hess


    def generate_hess(self,pop_tabs,indt,inp_tabs,mag_range,mag_step,mag_smooth,color_shift=0):
        r"""
        Hess diagram for the simple volumes. 
        
        Parameters:
        -----------
        - pop_tabs: dict
            Tables of populations with their keys.
        - indt: 1d-array
            Age indices for thin-disk population table. 
        - inp_tabs: dict
            Model predictions for potential, AVR, scale heights...
        - r_max: scalar
            Maximal radius of the spherical shell, pc.   
        - mag_range: list[list[scalar]]
            Minimal and maximal magnitude along the x- and y-axis of the Hess diagram, [[x_min,x_max],[y_min,y_max]]
        - mag_step: array-like
            Step along the x- and y-axis of the Hess diagram, [dx,dy]. 
        - mag_smooth: array-like
            Size of the window for smoothing, mag. 

        Returns:
        --------
        Hess diagram, 2d-array
        """
        
        sum_Nz, mag = [], [[], [], []]
        
        pop_tabs = self.pops_in_volume(pop_tabs,indt,inp_tabs)

        for table in pop_tabs.values():
            sum_Nz.extend(table['Nz'])
            mag = _extend_mag_(mag,table,['G_EDR3','G_EDR3','GRP_EDR3'])

        sum_Nz = np.array(sum_Nz)
        
        hess = self.hess_from_data(np.subtract(mag[1],mag[2])+color_shift,mag[0],
                                   mag_range,mag_step,weights=sum_Nz,mag_smooth=mag_smooth)
        
        return hess.T



class SFRHandler():

    def __init__(self, p, a, inp):
        self.a = a
        self.p = p
        self.inp = inp
    
    def create_reference_sfr(self):
        return {'d':self.inp['SFRd0'],'t':self.inp['SFRt0']}
    
    def sfrd_mcmc(self,**kwargs):

        # Get SFR parameters either from namedtuple p (default values)
        # or from kwargs (custom values)

        par_names = ['dzeta','eta','td1','td2','sigmad',
                     'sigmap0','sigmap1','tpk0','tpk1','dtp']
        
        pars = {par:np.nan for par in par_names}

        for name in list(pars.keys()):
            if name in kwargs:
                pars[name] = kwargs[name]
            else:
                if name in ['dzeta','eta','td1','td2','sigmad','dtp']:
                    pars[name] = getattr(self.p,name)
                elif name == 'sigmap0':
                    pars[name] = getattr(self.p,'sigmap')[0]
                elif name == 'sigmap1':
                    pars[name] = getattr(self.p,'sigmap')[1]
                elif name == 'tpk0':
                    pars[name] = getattr(self.p,'tpk')[0]
                elif name == 'tpk1':
                    pars[name] = getattr(self.p,'tpk')[1]

        sfr = SFR()
        SFRd0, NSFRd0, Fp0 = sfr.sfrd_sj21_multipeak(tp,
                                                     tr,
                                                     self.a.t,
                                                     pars['dzeta'],
                                                     pars['eta'],
                                                     pars['td1'],
                                                     pars['td2'],
                                                     pars['sigmad'],
                                                     np.array([pars['sigmap0'],pars['sigmap1']]),
                                                     np.array([pars['tpk0'],pars['tpk1']]),
                                                     pars['dtp'],
                                                     g=self.inp['gd0']
                                                     )   
        return SFRd0, NSFRd0, Fp0
    

    def sfrt_mcmc(self,**kwargs):

        par_names = ['gamma','beta','tt1','tt2','sigmat']
        
        pars = {par:np.nan for par in par_names}

        for name in list(pars.keys()):
            if name in kwargs:
                pars[name] = kwargs[name]
            else:
                pars[name] = getattr(self.p,name)

        sfr = SFR()

        SFRt0, NSFRt0 = sfr.sfrt_sj21(self.a.t[:self.a.jt],
                                      pars['gamma'],
                                      pars['beta'],
                                      pars['tt1'],
                                      pars['tt2'],
                                      pars['sigmat'],
                                      g=self.inp['gt']
                                      )
        
        return SFRt0, NSFRt0


    def update_sfr(self, **kwargs):

        # Star formation rate function based on the calculated mass loss function

        SFRd0, NSFRd0, Fp0 = self.sfrd_mcmc(**kwargs)
        SFRt0, NSFRt0 = self.sfrt_mcmc(**kwargs)
                                            
        SFRtot0 = np.concatenate((np.add(SFRd0[:self.a.jt],SFRt0),SFRd0[self.a.jt:]),axis=None)
        NSFRtot0 = SFRtot0/np.mean(SFRtot0)

        names = ['SFRd0','NSFRd0','Fp0','SFRt0','NSFRt0','SFRtot0','NSFRtot0']
        tables = [SFRd0, NSFRd0, Fp0, SFRt0, NSFRt0, SFRtot0, NSFRtot0]

        for name, table in zip(names,tables):
            self.inp[name] = table
        
        return self.inp, {'d':self.inp['SFRd0'],'t':self.inp['SFRt0']}


class IMFHandler:

    def __init__(self, p, mres=0.0001, m_step = 0.01):

        self.p = p

        self.M_low, self.M_up = 0.08, 100      # Msun, lower and upper IMF mass limits
        self.mres = mres
        self.m_step = m_step

        mass_bins = np.arange(self.M_low,self.M_up + self.m_step, self.m_step)
        self.mass_binsc = mass_bins[:-1] + self.m_step/2


    def imf_mcmc(self,**kwargs):

        par_names = ['a0', 'a1', 'a2', 'a3', 'm0', 'm1', 'm2']
                     
        pars = {par:np.nan for par in par_names}

        for name in list(pars.keys()):
            if name in kwargs:
                pars[name] = kwargs[name]
            else:
                pars[name] = getattr(self.p,name)

        imf = IMF(self.M_low, self.M_up, mres = self.mres)  
        _, _ = imf.BPL_4slopes(pars['a0'],
                               pars['a1'],
                               pars['a2'],
                               pars['a3'],
                               pars['m0'],
                               pars['m1'],
                               pars['m2']
                               )
        return imf


    def create_reference_imf(self):

        imf_ref = IMF(self.M_low,self.M_up,mres=self.mres)        # Here we create class instance
        _, _ = imf_ref.BPL_4slopes(self.p.a0,self.p.a1,self.p.a2,self.p.a3,
                                         self.p.m0,self.p.m1,self.p.m2)   # and define the IMF 

        IMF_ref = [imf_ref.number_stars(mass - self.m_step/2, mass + self.m_step/2) for mass in self.mass_binsc] 

        return imf_ref, (self.mass_binsc, IMF_ref)
    

    def update_imf(self, **kwargs):

        imf = self.imf_mcmc(**kwargs)
        IMF_new = [imf.number_stars(mass - self.m_step/2, mass + self.m_step/2) for mass in self.mass_binsc]

        return imf, (self.mass_binsc, IMF_new)


class IFMRHandler():

    def __init__(self,extend_mf_limits=False):

        self.M_sun = 1 # all in solar units

        # Cummings+18 IFMR
        self.segments_cummings = {
            'padova':[ # Padova-calibrated
                {"range": [0.87, 2.80], "a": 0.0873, "a_err": 0.0190, "b": 0.476, "b_err": 0.033},
                {"range": [2.80, 3.65], "a": 0.181, "a_err": 0.041, "b": 0.210, "b_err": 0.131},
                {"range": [3.65, 8.20], "a": 0.0835, "a_err": 0.0144, "b": 0.565, "b_err": 0.073},
            ],
            'mist': [ # MIST-calibrated
                {"range": [0.83, 2.85], "a": 0.080, "a_err": 0.016, "b": 0.489, "b_err": 0.030},
                {"range": [2.85, 3.60], "a": 0.187, "a_err": 0.061, "b": 0.184, "b_err": 0.199},
                {"range": [3.60, 7.20], "a": 0.107, "a_err": 0.016, "b": 0.471, "b_err": 0.077},
            ]
        }

        # Calculate segments' ranges in Mf for the inverted IFMR
 
        for cal in ['padova','mist']:
            for i, seg in enumerate(self.segments_cummings[cal]):

                vals, sigmas = self.cummings(seg['range'],calibration=cal)

                if extend_mf_limits:
                    if i==0:
                        val_upper = vals[1]
                        val_lower = vals[0] - 3*sigmas[0]
                        if val_lower < 0:
                            val_lower = 0 
                    elif i==len(self.segments_cummings[cal]) - 1:
                        val_lower = vals[0]
                        val_upper = vals[1] + 3*sigmas[1]
                    else:
                        val_lower, val_upper = vals
                else:
                    val_lower, val_upper = vals
                
                seg['range_r'] = np.round([val_lower,val_upper],2)


    def update_ifmr(self,calibration='padova',**kwargs):

        for par in list(kwargs.keys()):
            if par == 'm_br1':
                self.segments_cummings[calibration][0]['range'][1] = kwargs[par]
                self.segments_cummings[calibration][1]['range'][0] = kwargs[par]
            if par == 'm_br2':
                self.segments_cummings[calibration][1]['range'][1] = kwargs[par]
                self.segments_cummings[calibration][2]['range'][0] = kwargs[par]
            if par == 'alpha1':
                self.segments_cummings[calibration][0]['a'] = kwargs[par]
            if par == 'alpha2':
                self.segments_cummings[calibration][1]['a'] = kwargs[par]
            if par == 'alpha3':
                self.segments_cummings[calibration][2]['a'] = kwargs[par]

        for i, seg in enumerate(self.segments_cummings[calibration]):
            vals, _ = self.cummings(seg['range'],calibration=calibration)
            val_lower, val_upper = vals
            seg['range_r'] = np.round([float(val_lower),float(val_upper)],2)

        # Ensure continuity by updating offsets
        b1 = self.segments_cummings[calibration][0]['b']
        a1 = self.segments_cummings[calibration][0]['a']
        a2 = self.segments_cummings[calibration][1]['a']
        a3 = self.segments_cummings[calibration][2]['a']
        m_br1 = self.segments_cummings[calibration][0]['range'][1]
        m_br2 = self.segments_cummings[calibration][1]['range'][1]

        self.segments_cummings[calibration][1]['b'] = float(np.round(b1 - m_br1*(a2 - a1),3))
        self.segments_cummings[calibration][2]['b'] = float(np.round(b1 - m_br1*(a2 - a1) - m_br2*(a3 - a2),3))


    def cummings(self,Mini,calibration='padova'):

        segment = self.segments_cummings[calibration]

        x = Mini
        if np.isscalar(Mini):
            x = [Mini]

        Mf, sigma_Mf = [], [] 
        for mini in x:
            if mini < segment[0]['range'][0] or mini > segment[-1]['range'][-1]:
                Mf.append(np.nan)
                sigma_Mf.append(np.nan)
            else:
                # Find the right segment
                for seg in segment:
                    if (seg['range'][0] <= mini) and (mini <= seg['range'][1]):
                        mf = seg["a"] * mini + seg["b"] * self.M_sun
                        sigma_mf = np.sqrt((mini * seg["a_err"])**2 + (self.M_sun * seg["b_err"])**2)
                        Mf.append(mf)
                        sigma_Mf.append(sigma_mf)
                        break

        output = ((Mf[0], sigma_Mf[0]) if np.isscalar(Mini) else (np.array(Mf), np.array(sigma_Mf)))
        return output

    def cummings_r(self,Mf,calibration='padova'):

        segment = self.segments_cummings[calibration]

        x = Mf
        if np.isscalar(Mf):
            x = [Mf]
        
        Mini, sigma_Mini = [], []
        for mf in x:
            if mf < segment[0]['range_r'][0] or mf > segment[-1]['range_r'][-1]:
                Mini.append(np.nan)
                sigma_Mini.append(np.nan)
            else:
                for seg in segment:
                    if (seg['range_r'][0] <= mf) and (mf <= seg['range_r'][1]):
                        mini = (mf - seg["b"] * self.M_sun)/seg["a"]
                        sigma_mini = np.sqrt(((mf - seg["b"]*self.M_sun)/seg["a"] * seg["a_err"])**2 +\
                                            (self.M_sun * seg["b_err"])**2) / seg["a"]
                        
                        # Check for unphysical results due to extension of applicability range
                        if mini < 0.08:
                            mini = np.nan
                            sigma_mini = np.nan
                        Mini.append(mini)
                        sigma_Mini.append(sigma_mini)
                        break

        output = ((Mini[0], sigma_Mini[0]) if np.isscalar(Mf) else (np.array(Mini), np.array(sigma_Mini)))
        return output
    

    def apply_ifmr_scatter(self,N_ref, Mini, sigma_Mini):
        """Returns new number desities column N_new calculated with IFMR scatter.
        Input arrays: 
        - reference N_ref (corresponding to deterministic IFMR)
        - Mini
        - sigma_Mini
        """
        
        n  = len(Mini)
        N_new = np.zeros_like(N_ref, dtype=float)

        # Precompute bin centers and widths
        Mini_edges = np.zeros((n + 1))
        Mini_edges[0], Mini_edges[-1] = Mini[0], Mini[-1]
        Mini_edges[1:-1] = [np.mean([Mini[i],Mini[i+1]]) for i in np.arange(n - 1)]

        # Precompute edges
        # compute integrated Gaussian probabilities for each destination bin
        # bin integral = CDF((b-mu)/sigma) - CDF((a-mu)/sigma)
        a_edges = Mini_edges[:-1]
        b_edges = Mini_edges[1:]

        # For all j, compute CDF values in one go:
        cdf_a = norm.cdf((a_edges[None,:] - Mini[:,None]) / sigma_Mini[:,None])
        cdf_b = norm.cdf((b_edges[None,:] - Mini[:,None]) / sigma_Mini[:,None])
        bin_integrals = cdf_b - cdf_a
        
        # Apply scatter
        for j in range(n):
            mu = Mini[j]
            sigma = sigma_Mini[j]

            # destination bin indices within ±3σ
            nsig = 3
            lo = mu - nsig * sigma
            hi = mu + nsig * sigma
            k0 = max(np.searchsorted(Mini_edges, lo) - 1, 0)
            k1 = min(np.searchsorted(Mini_edges, hi), n)

            if k0 >= k1:
                continue

            # bin weights for row j
            bin_weights = bin_integrals[j, k0:k1]
            #bin_weights /= bin_weights.sum()

            # accumulate
            N_new[j] += np.sum(N_ref[k0:k1] * bin_weights)

        return N_new


    def add_sigma_mini_column(self,pop_tabs,ind_wds,calibration='padova'):

        labels = list(pop_tabs.keys())

        for label in labels:

            tab = pop_tabs[label]
            ind_wd = ind_wds[label]

            tab['sigma_Mini'] = [np.nan for _ in np.arange(len(tab['Mini']))] 
            for ind in ind_wd:
                _, sigma_Mini = self.cummings_r(tab['Mf'][ind],calibration=calibration)                            
                tab['sigma_Mini'][ind] = sigma_Mini

        return pop_tabs
    

    def extract_mini_grid(self,pop_tabs, ind_wd):
        '''
        Returns unique values of Mini and sigma_Mini for WD populations 
        '''
        mini_all = []
        sigma_all = []

        for tab_key in pop_tabs:
            tab = pop_tabs[tab_key]
            inds_list = ind_wd[tab_key]
            for ind in inds_list:
                mini_all.append(tab['Mini'][ind])
                sigma_all.append(tab['sigma_Mini'][ind])

        mini_all = np.concatenate(mini_all)
        sigma_all = np.concatenate(sigma_all)

        # extract unique Mini and corresponding sigma by first occurrence
        mini_grid, idx = np.unique(mini_all, return_index=True)
        sigma_grid = sigma_all[idx]

        return mini_grid, sigma_grid
    

    def apply_ifmr_scatter_poptabs_exact(self,pop_tabs,ind_wd):
        '''
        IFMR scatter -- exact approach. 
        Returns updated tables with a new column N_sm
        '''

        dt = 0.025  # time/age resolution
        jd = 520    # number of single-age populations

        tables = pop_tabs.copy()

        mini_grid, mini_sigma_grid = self.extract_mini_grid(pop_tabs,ind_wd)

        # Iterate over thin/ thick disk and halo 
        for tab,indices in zip(tables.values(),ind_wd.values()): 
            
            # Initialize the new column with deterministic values
            tab['N_sm'] = tab['N']

            # Iterate over ('DA', 'DB') WD types
            for ind in indices: 
                
                indt = np.array(np.round(tab['age'][ind],3)//dt,dtype=int)

                # Iterate over all ages
                for j in range(jd):
                    
                    # Find all rows with this age
                    ind_age = np.where(indt==j)[0]
                    rows = ind[ind_age]
                    
                    # Sum over all masses for thi age
                    indm = np.searchsorted(mini_grid, tab['Mini'][rows])
                    N_sum = np.bincount(indm, weights=tab['N'][rows], minlength=len(mini_grid))

                    # Apply IFMR scatter
                    N_sum_sm = self.apply_ifmr_scatter(N_sum,mini_grid,mini_sigma_grid)

                    # Redistribute proportional to original N in that age bin
                    for i in range(len(mini_grid)):
                        mask = indm == i
                        if N_sum[i] > 0:
                            tab['N_sm'][rows[mask]] = N_sum_sm[i] * tab['N'][rows[mask]] / N_sum[i]
                        else:
                            tab['N_sm'][rows[mask]] = 0

                    # Renormalize to keep total number the same?
                    #if sum(tab['N_sm'][rows]) > 0:
                    #    norm_corr = sum(tab['N'][rows])/sum(tab['N_sm'][rows])
                    #    tab['N_sm'][rows] *= norm_corr

        return tables
    

    def apply_ifmr_scatter_poptabs_reduced(self,pop_tabs,ind_wd):
        '''
        IFMR scatter -- reduced approach. 
        Returns updated tables with a new column N_sm
        '''
            
        tables = pop_tabs.copy()

        # Get unique values of initial masses and its uncertainties
        mini_grid, mini_sigma_grid = self.extract_mini_grid(pop_tabs,ind_wd)

        # Iterate over thin/ thick disk and halo
        for tab,indices in zip(tables.values(),ind_wd.values()):  
            
            tab['N_sm'] = tab['N']

            # Iterate over ('DA', 'DB') WD types
            for ind in indices: 
                
                rows = ind
                
                # Sum all rows with this mass (across different ages!)
                indm = np.searchsorted(mini_grid, tab['Mini'][rows])
                N_sum = np.bincount(indm, weights=tab['N'][rows], minlength=len(mini_grid))
                N_sum_sm = self.apply_ifmr_scatter(N_sum,mini_grid,mini_sigma_grid)

                # Redistribute proportional to original N
                for i in range(len(mini_grid)):
                    mask = indm == i
                    if N_sum[i] > 0:
                        tab['N_sm'][rows[mask]] = N_sum_sm[i] * tab['N'][rows[mask]] / N_sum[i]
                    else:
                        tab['N_sm'][rows[mask]] = 0

                # Renormalize to keep total number the same?
                #if sum(tab['N_sm'][rows]) > 0:
                #    norm_corr = sum(tab['N'][rows])/sum(tab['N_sm'][rows])
                #    tab['N_sm'][rows] *= norm_corr

        return tables
    

class PopHandler:

    def __init__(self, p, a, inp):
        self.a = a
        self.p = p
        self.inp = inp

    def get_age_mass_idx(self, pop_tabs, mass_binsc):

        indt = {}
        indm = {}

        indt['d'] = np.array([np.argmin(np.abs(time - self.a.t)) for time in pop_tabs['d']['t']])
        indm['d'] = np.array([np.argmin(np.abs(mass - mass_binsc)) for mass in pop_tabs['d']['Mini']])

        indt['t'] = np.array([np.argmin(np.abs(time - self.a.t[:self.a.jt])) for time in pop_tabs['t']['t']])
        indm['t'] = np.array([np.argmin(np.abs(mass - mass_binsc)) for mass in pop_tabs['t']['Mini']])

        indm['sh'] = np.array([np.argmin(np.abs(mass - mass_binsc)) for mass in pop_tabs['sh']['Mini']])

        return indt, indm


    def get_wd_idx(self, pop_tabs):
        ind_da, ind_db = {}, {}
        for label in ['d','t','sh']:
            ind_da[label] = np.where(pop_tabs[label]['phase']==10)[0]
            ind_db[label] = np.where(pop_tabs[label]['phase']==11)[0]
        return (ind_da, ind_db)


    def display_wd_stats(self,mode_pop,ind_wd):
        print('\n')
        if mode_pop in ('wd','tot'):
            print('WD indices d:\t',len(ind_wd['d'][0]),len(ind_wd['d'][1]))
            print('WD indices t:\t',len(ind_wd['t'][0]),len(ind_wd['t'][1]))
            print('WD indices sh:\t',len(ind_wd['sh'][0]),len(ind_wd['sh'][1]))
    

    def make_wd_idx_dict(self, pop_tabs):

        ind_da, ind_db = self.get_wd_idx(pop_tabs)
        ind_wd = {label: [ind_da[label],ind_db[label]] for label in pop_tabs.keys()}
        return ind_wd


    def create_reference_pop_tabs(self, imf_ref, mode_iso, **kwargs):

        # SA are synthesized using multiprocessing, so stellar_assemblies_r() must be wrapped like this:
        stellar_assemblies_r(self.p.Rsun,self.p,self.a,
                             self.inp['AMRd0'],self.inp['AMRt'],
                             self.inp['SFRd0'],self.inp['SFRt0'],
                             self.p.sigmash,imf_ref.number_stars,mode_iso,3,**kwargs)

        pop_tabs = {}
        for pop in ['d','t','sh']:
            pop_tabs[pop] = Table.read(os.path.join(self.a.T['poptab'],
                                                    ''.join(('SSP_R',str(self.p.Rsun),'_' + pop +'_',mode_iso,'.csv'))))
            pop_tabs[pop]['N_ref'] = np.copy(pop_tabs[pop]['N'])
            pop_tabs[pop]['t'] = tp - pop_tabs[pop]['age'] 
        
        return pop_tabs
    

    def create_reference_columns(self, pop_tabs, colnames_ref):
        for label in pop_tabs.keys():
            for name in colnames_ref:
                pop_tabs[label][f'{name}_ref'] = pop_tabs[label][name]

        return pop_tabs


    def reset_columns(self,pop_tabs):

        for col in ['N', 'Mini', 'age', 'age_WD']: 
            for label in pop_tabs.keys():
                pop_tabs[label][col] = pop_tabs[label][f'{col}_ref']

        return pop_tabs
    

    def update_pop_tabs(self, opt_params, pop_tabs, **kwargs):
        
        pop_tabs = self.reset_columns(pop_tabs)

        labels = pop_tabs.keys()
        opt_classes = opt_params.keys()

        indt = kwargs['indt'] # read in any case
        SFR_ref = kwargs['SFR_ref']
        #if any(x in opt_classes for x in ['ifmr', 'dcool', 'sfr']):

        if any(x in opt_classes for x in ['ifmr', 'f_dadb', 'dcool']):
            ind_wd = kwargs['ind_wd']

            time_grid = np.round(self.a.t,3)
            min_age = {'d':0,'t':13 - self.p.tt2}
            max_time_idx = {'d':self.a.jd - 1, 't':self.a.jt - 1}
        
        if 'sfr' in opt_classes:
            SFR_new = kwargs['SFR_new']
        
        if 'imf' in opt_classes:
            IMF_new = kwargs['IMF_new']
            IMF_ref = kwargs['IMF_ref']
            indm = kwargs['indm']
            imf_weights = np.array([p_new/p_old for p_new,p_old in zip(IMF_new,IMF_ref)])
                                        

        for label in labels:

            if any(x in opt_classes for x in ['ifmr', 'f_dadb', 'dcool']):
                ind_da, ind_db = ind_wd[label]
                all_wd_idx = np.concatenate(ind_wd[label])
                age_wd_ref_arr = pop_tabs[label]['age_WD_ref'][all_wd_idx]
                age_tot_ref_arr = pop_tabs[label]['age'][all_wd_idx]

            # -------- IFMR: vectorize if present ----------
            if 'ifmr' in opt_classes and label != 'sh':

                # get slices / cached arrays
                Mf_arr = pop_tabs[label]['Mf'][all_wd_idx]
                FeH_arr = pop_tabs[label]['FeH'][all_wd_idx]

                # Call ifmr.cummings_r s on the whole array
                ifmr_handler = kwargs['ifmr_handler']
                ifmr_handler.update_ifmr(**opt_params['ifmr'])

                # Update initial masses and write back in one go
                Mini_new, _ = ifmr_handler.cummings_r(Mf_arr, calibration='padova')
                pop_tabs[label]['Mini'][all_wd_idx] = Mini_new

                # compute MS ages via prebuilt interpolator (vectorized)
                msage_handler = kwargs['msage_handler']
                points = np.column_stack([FeH_arr, Mini_new])
                age_ms = msage_handler.interp_age_ms(points)      # vector
                age_tot_wd = age_ms + age_wd_ref_arr       # updated total ages
            elif 'dcool' in opt_classes and label != 'sh':
                # If no IFMR, start from stored age_tot (which may be age_ref)
                age_tot_wd = age_tot_ref_arr.copy()


            # -------- dcool: apply cooling delay correction (vectorized) ----------
            if 'dcool' in opt_classes and label != 'sh':
                alpha_cool = opt_params['dcool']['alpha_cool']
                age_tot_wd = age_tot_wd + alpha_cool * age_wd_ref_arr

            
            if (('ifmr' in opt_classes) or ('dcool' in opt_classes)) and (label != 'sh'):
                # mask invalid (outside [min_time, 13]) or non-finite
                bad_mask = (~np.isfinite(age_tot_wd)) | (age_tot_wd > 13.0) | (age_tot_wd < min_age[label])
                good_mask = ~bad_mask

                # Compute new time index array and clip to a valid range
                indt_new = np.searchsorted(time_grid, 13.0 - age_tot_wd)
                indt_new = np.clip(indt_new, 0, max_time_idx[label])
                # update indt in place
                indt[label][all_wd_idx] = indt_new

                # indt_ref for WD rows
                indt_ref = indt[label][all_wd_idx]   

                # Compute SFR weights only for valid rows
                # (correction for the total age bin shift)
                if np.any(good_mask):
                    sfr_ref_arr = SFR_ref[label]                         
                    # gather numerator and denominator vectors
                    num = sfr_ref_arr[indt_new[good_mask]]
                    den = sfr_ref_arr[indt_ref[good_mask]]
                    # avoid div by zero
                    den = np.where(den == 0, 1e-30, den)
                    sfr_shift_weights = num / den
                    # apply weights back into pop_tabs N
                    pop_tabs[label]['N'][all_wd_idx[good_mask]] *= sfr_shift_weights
                    
                # zero out invalid rows
                if np.any(bad_mask):
                    pop_tabs[label]['N'][all_wd_idx[bad_mask]] = 0
                    indt[label][all_wd_idx[bad_mask]] = 0  # formal index


            # -------- f_dadb: account for change in DA/DB fraction ----------
            # Apply to 'd', 't', 'sh'
            if 'f_dadb' in opt_classes:
                f_da = opt_params['f_dadb']['f_da']
                f_da_teff = kwargs.get('f_da_teff', None)
            
                da_weight = f_da/0.8
                db_weight = (1 - f_da)/0.2

                teff_factor_da, teff_factor_db = 1, 1

                # Check for dependence on Teff
                if f_da_teff:
                    teff_factor_db = fdb_parabola(10**(pop_tabs[label]['logT'] - 3))
                    teff_factor_da = (1 - teff_factor_db)
                
                pop_tabs[label]['N'][ind_da] *= da_weight*teff_factor_da
                pop_tabs[label]['N'][ind_db] *= db_weight*teff_factor_db


            # -------- SFR: apply SFR weights ----------
            # Apply to 'd', 't' only
            if 'sfr' in opt_classes and label != 'sh':

                sfr_weights = SFR_new[label]/SFR_ref[label]
                pop_tabs[label]['N'] *= sfr_weights[indt[label]]


            # -------- IMF: apply IMF weights ----------
            # Apply to 'd', 't', 'sh'
            if 'imf' in opt_classes:
                pop_tabs[label]['N'] *= imf_weights[indm[label]]

        return pop_tabs, indt
    

def extract_model_tables(out,inp):

    return {'Fi':out['phi'],'Fp':inp['Fp0'],'AVR':out['avr'],
            'Hd':out['hd'],'Hdp':out['hdp'],'Ht':out['ht'],'Hsh':out['hsh']}
        

def fdb_parabola(Teff):
    # Teff in 10^3 format
    parabola = lambda x, a, b, c: a*x**2 + b*x + c
    parameters = [1.4e-4, -1.15983436e-2, 3.11929944e-1]
    return parabola(Teff,*parameters)


def piecewise_linear_v2(x, x0, b, k1, k2):
    '''2-slope piecewise linear function with breakpoint at x0.'''
    condlist = [x < x0, x >= x0]
    funclist = [lambda x:k1*x + b, lambda x:k2*(x - x0) + b + k1*x0]
    return np.piecewise(x,condlist,funclist)


def piecewise_linear_v3(x, x0, x1, b, k1, k2, k3):
    '''3-slope piecewise linear function with breakpoints at x0 and x1.'''
    condlist = [x < x0, (x >= x0) & (x < x1), x >= x1]
    funclist = [lambda x: k1*x + b, 
                lambda x: k2*(x - x0) + b + k1*x0, 
                lambda x: k3*(x - x1) + b + k1*x0 + k2*(x1 - x0)]
    return np.piecewise(x, condlist, funclist)


def chi_square(y1, y2):
    '''Chi-square normalized to the number of points.'''
    return sum((y1 - y2)**2)/len(y1)


class MSAgeHandler():

    def __init__(self,feh=None,param_file=None):

        self.feh_grid = []
        self.params = []

        try:
            with open(param_file,'r') as f:
                for line in f:
                    if line[0] not in ['#','\n']:
                        values = line.split(' ')
                        self.feh_grid.append(float(values[0]))
                        parameter_set = [float(val) for val in values[1:] if val not in ('','\n')]
                        n_parameters = len(parameter_set)
                        if n_parameters == 4:
                            parameter_set[0] = np.log10(parameter_set[0]) # convert Mbr to log
                        if n_parameters == 5:
                            parameter_set[0] = np.log10(parameter_set[0])
                            parameter_set[1] = np.log10(parameter_set[1])
                        self.params.append(parameter_set)
        except:
            pass

        self.feh_grid = np.array(self.feh_grid)

        self.pwl = ['pwl2' if len(par)==4 else 'pwl3' for par in self.params]
        self.pwl_funcs = {'pwl2':piecewise_linear_v2,'pwl3':piecewise_linear_v3}

        self.set_feh(feh)       

        # Tabulate and create an interpolator 
        if self.params != []:
            _ = self.build_interpolator()


    def set_feh(self,feh):
        self.feh = feh
        self.feh_idx = (np.argmin(np.abs(self.feh_grid - self.feh)) if self.feh!=None else None)
        return self.feh_idx
    

    def get_age_ms(self,mass_ini,feh=None):
        
        if not feh and not self.feh and not self.feh_idx:
            raise ValueError('Parameter feh (metallicity [Fe/H]) must be given!')
        elif feh:
            self.set_feh(feh)

        func = self.pwl_funcs[self.pwl[self.feh_idx]]

        if np.isscalar(mass_ini):
            power = func(np.log10(mass_ini),*self.params[self.feh_idx]) if not np.isnan(mass_ini) else np.nan
        else:
            power = np.array([func(np.log10(mass),*self.params[self.feh_idx]) if not np.isnan(mass) else np.nan for mass in mass_ini])
        tau_ms = 10**power  # in Gyr

        return tau_ms
    

    def compute_age_ms_grid(self,log_mini_grid=np.linspace(np.log10(0.08),np.log10(8.2),100)):

        self.log_mini_grid = log_mini_grid
        feh_grid = self.feh_grid

        # Compute lifetime table
        age_ms_grid = np.zeros((len(feh_grid), len(log_mini_grid)))
        for i, feh in enumerate(feh_grid):
            for j, mini in enumerate(log_mini_grid):
                age_ms_grid[i, j] = self.get_age_ms(10**mini, feh=feh)

        return age_ms_grid
    

    def build_interpolator(self):

        age_ms_grid = self.compute_age_ms_grid()

        interp_age_ms = RegularGridInterpolator(
                            (self.feh_grid, self.log_mini_grid),
                            age_ms_grid,
                            bounds_error=False,
                            fill_value=None
                        )
        
        self.interp_age_ms = interp_age_ms
        return interp_age_ms
    

    def retrieve_ms_lifetimes(self,FeH_grid,dir_out='MS_lifetime',
                              iso_dir='jjmodel/input/isochrones/Padova/multiband'):

        # Create dict for output data
        age_ms_data = {str(feh):{'mass':[],'age':[]} for feh in FeH_grid}

        # Iterate over metallicities and isochrone files
        for feh in FeH_grid:

            feh_dir = os.path.join(iso_dir,'iso_feh' + str(feh))
            iso_files = os.listdir(feh_dir)

            # For all ages in the metallicity directory
            for iso_file in iso_files:

                # Read isochrone
                iso_data = np.loadtxt(os.path.join(feh_dir,iso_file)).T

                # Get age from the filename
                iso_age = float(iso_file.split('age')[1].split('.txt')[0]) # in Gyr

                # Get turn-off mass (lowest mass with label=2)
                # 0-th column = initial mass
                # 5-th column = evolutionary stage (1 = MS, 2 = SGB)
                turnoff_mass = iso_data[0][np.where(iso_data[5] == 2)[0]][0]

                # Append to the dict
                age_ms_data[str(feh)]['mass'].append(turnoff_mass)
                age_ms_data[str(feh)]['age'].append(iso_age)

            # Convert lists to arrays and format output 
            output = np.array([age_ms_data[str(feh)]['mass'], age_ms_data[str(feh)]['age']])
            ind_u = np.unique(output[0],return_index=True)[1]

            masses = output[0][ind_u]
            ages = [min(output[1][np.where(output[0]==output[0][i])[0]]) for i in ind_u]
            
            output = np.array([masses,ages])
            np.savetxt(os.path.join(dir_out,''.join(('fe'+str(feh)+'.txt'))), 
                    output.T, header='Mini[M_sun]\tAge[Gyr]')
            

    def plot_age_ms(self, mets, dir_data='MS_lifetime', dir_out='MS_lifetime/plots'):
        '''
        Visualization of the MS-lifetime data for a list of metallicities.
        '''

        fig, ax = plt.subplots(figsize=(8,6))
        cmap = mpl.colormaps['RdBu']

        # Create a Normalize instance based on your metallicity values
        norm = mpl.colors.Normalize(vmin=-2.5, vmax=0.5)
        # Map each metallicity value through the colormap using the normalization
        colors = cmap(norm(mets))
        # Take colors at regular intervals spanning the colormap.
        colors = cmap(np.linspace(0, 1, len(mets)))
        
        # Plot data
        for met, c in zip(mets,colors):

            data = np.loadtxt(os.path.join(dir_data,''.join(('fe'+str(met)+'.txt')))).T
            mass = data[0]
            age = data[1]
            x = np.log10(mass)
            y = np.log10(age)

            ax.loglog(10**x, 10**y, "x", ms=3, color=c)
            ax.set_xlabel('M_initial [Msun]')
            ax.set_ylabel('MS lifetime [Gyr]')

        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

        # Pass both cmap and norm to ScalarMappable so the colorbar matches your data
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # required for older matplotlib
        cax = fig.colorbar(sm, ax=ax)
        cax.set_label('[Fe/H]')

        plt.savefig(os.path.join(dir_out,'MS_lifetimes_allFeH.png'))
        
        return fig, ax
    

    def fit_ms_lifetime(self, met, func='v1', Mbr = None, 
                    dir_data='MS_lifetime', dir_out='MS_lifetime/plots', 
                    plot=False, stat=True, verbose=True):
        '''
        Returns fit function and parameters for the MS-lifetime relation for a given metallicity.
        The fit is done in log-log space.
        Parameter func can be 'v1', 'v2' or 'v3' for 2-slope (free or fixed breakpoint),
        or a 3-slope piecewise linear function, respectively.
        If plot=True, a plot of the data, fit and residuals is shown.
        '''

        # Get the data
        data = np.loadtxt(os.path.join(dir_data,''.join(('fe'+str(met)+'.txt')))).T

        mass = data[0]
        age = data[1]
        x = np.log10(mass)
        y = np.log10(age)
        
        # Fit options
        param = {'v1':{'func': lambda x, b, k1, k2: piecewise_linear_v2(x, np.log10(Mbr), b, k1, k2),
                    'p0': [0.5, -3.5, -2.5],
                    'bounds': ((-10, -5, -5),(10, -1.0, -1.0))
                    },
                'v2':{'func': piecewise_linear_v2, 
                    'p0': [0.04, 0.5, -3.5, -2.5],
                    'bounds': ((0.02, -10, -5, -5),(0.35, 10, -1.0, -1.0))
                    },
                'v3':{'func': piecewise_linear_v3, 
                    'p0': [0.06, 0.12, 1, -3.1, -3.1, -3.1],
                    'bounds': ((0.02, 0.1, -10, -5, -5, -5),(0.12, 0.35, 10, -1.0, -1.0, -1.0))
                    }
                }
        
        # Choose the fit function 
        piecewise_linear = param[func]['func']

        # Perform the fit
        p , _ = optimize.curve_fit(piecewise_linear, x, y, 
                                p0 = param[func]['p0'], 
                                bounds = param[func]['bounds'],
                                #sigma = 1/(x + 1e-12) # weights
                                )
        
        # Output dict for miscellaneous information
        msl = {}

        if stat:
            chi2 = round(chi_square(y, piecewise_linear(x, *p)),5)
            msl['chi2'] = chi2

            if verbose:
                print('Chi^2 = ', chi2)
                if func == 'v1':
                    print('M_br1 = ', round(Mbr,2), 'Msun')
                    print('alpha1 = ', round(p[1],2))
                    print('alpha2 = ', round(p[2],2))
                if func == 'v2':
                    print('M_br1 = ', round(10**p[0],2), 'Msun')
                    print('alpha1 = ', round(p[2],2))
                    print('alpha2 = ', round(p[3],2))
                if func == 'v3':
                    print('M_br1 = ', round(10**p[0],2), 'Msun')
                    print('M_br2 = ', round(10**p[1],2), 'Msun')
                    print('alpha1 = ', round(p[3],2))
                    print('alpha2 = ', round(p[4],2))
                    print('alpha3 = ', round(p[5],2))

        if plot:
            f, ax = plt.subplots(2,1, figsize=(6,8),sharex=True)
            ax[0].loglog(10**x, 10**y, "x", ms=5, c='steelblue',label='Data from isochrones')
            ax[0].loglog(10**x, 10**piecewise_linear(x, *p),c='tomato',label='Fitted function')
            #ax[0].loglog(10**x, 10**piecewise_linear(x, *np.round(p,2)),c='orange')
            kwg = {'ls':'--', 'color':'grey','lw':0.5,'zorder':10}
            if func == 'v1':
                if not Mbr:
                    raise ValueError("For func='v1', Mbr must be provided.")
                ax[0].vlines(Mbr, 0.02, 20,**kwg)
            elif func == 'v2':
                ax[0].vlines(10**p[0], 0.02, 20, **kwg)
            elif func == 'v3':
                ax[0].vlines(10**p[0], 0.02, 20, **kwg)
                ax[0].vlines(10**p[1], 0.02, 20, **kwg)
            ax[0].set_xlim(0.7,10)
            ax[0].set_ylabel('Age [Gyr]')
            ax[0].legend(loc=1)
            
            ax[1].scatter(10**x, 10**(y - piecewise_linear(x, *p)) - 1, s=1)
            if stat:
                chi2 = round(chi_square(y, piecewise_linear(x, *p)),5)
                ax[1].text(0.75, 0.1, r'$\mathrm{\chi^2 \, = \, }$'+ str(chi2), transform=ax[1].transAxes)
            ax[1].set_xticks(np.arange(1,10))
            ax[1].set_xticklabels(np.arange(1,10))
            ax[1].set_ylabel('Age [Gyr]')
            ax[1].set_xlabel('Mass [Msun]')

            if func == 'v1':
                figname = 'MS_lifetime_fit_fe' + str(met) + '_' + func + '_Mbr' + str(Mbr) + '.png'
            else:
                figname = 'MS_lifetime_fit_fe' + str(met) + '_' + func + '.png'

            plt.savefig(os.path.join(dir_out,figname))

            msl['fig'] = (f, ax)
        
        return (piecewise_linear, p), msl
    

    def fit_all_metallicities(self,FeH_grid, func='v1', Mbr=None, dir_data='MS_lifetime', 
                              dir_out='MS_lifetime', plot=True, stat=True):
        '''
        Fit the MS-lifetime data for all metallicities in FeH_grid.
        The fit is done in log-log space.
        Parameter func can be 'v1', 'v2' or 'v3' for 2-slope (free or fixed breakpoint),
        or a 3-slope piecewise linear function, respectively.
        The fit parameters are saved to a text file.
        '''

        msl1 = {}

        if stat:
            chi2_array = []

        if func == 'v1' or func == 'v2':
            labels = ['FeH', 'M_br1', 'b', 'alpha1', 'alpha2']
        elif func == 'v3':
            labels = ['FeH', 'M_br1', 'M_br2', 'b', 'alpha1', 'alpha2', 'alpha3']

        header = ''.join((['{:<10}'.format(l) for l in labels]))
        print(header)

        output = {key:[] for key in labels}

        filename = 'tau_ms_params_' + func + '.txt'
        if func == 'v1':
            filename = 'tau_ms_params_' + func + '_Mbr' + str(Mbr) + '.txt'

        with open(os.path.join(dir_out,filename),"w") as f:
            f.write('# M_br in Msun, transform to log10 before passing to the fit function!\n')
            f.write('# ' + header + '\n')
            for feh in FeH_grid:
                with warnings.catch_warnings(action='ignore'):
                    (_, p), msl2 = self.fit_ms_lifetime(feh,func=func,Mbr=Mbr,
                                                dir_data=dir_data,plot=False,stat=stat,verbose=False);
                    if stat:
                        chi2_array.append(msl2['chi2'])
                if func=='v1':
                    params = [feh, Mbr, *p]
                elif func=='v2':
                    params = [feh, 10**p[0], *p[1:]]
                elif func=='v3':
                    params = [feh, 10**p[0], 10**p[1], *p[2:]]
                
                for key in labels:
                    output[key].append(params[labels.index(key)])
                
                line = ''.join((['{:<10}'.format(str(round(pa,2))) for pa in params]))
                print(line)
                f.write(line + '\n')

            if stat:
                chi2_mean = round(np.mean(chi2_array),5)
                chi2_std = round(np.std(chi2_array),5)
                print('\nMean chi^2 = ', chi2_mean, ' +/- ', chi2_std)
                f.write('\n# Mean chi^2 = ' + str(chi2_mean) + ' +/- ' + str(chi2_std) + '\n')

                msl1['chi2_mean'] = chi2_mean
                msl1['chi2_std'] = chi2_std

        if plot:

            lbs = {'M_br1':{'c':'m','l':r'$\mathrm{M_{br1}}$'},
                'M_br2':{'c':'violet','l':r'$\mathrm{M_{br2}}$'}, 
                'b':{'c':'orange','l':r'$\mathrm{b}$'},
                'alpha1':{'c':'lawngreen','l':r'$\mathrm{\alpha_{1}}$'},
                'alpha2':{'c':'green','l':r'$\mathrm{\alpha_{2}}$'},
                'alpha3':{'c':'blue','l':r'$\mathrm{\alpha_{3}}$'}
                }

            f, ax = plt.subplots(len(params)-1,1,figsize=(8,6))
            ax[0].set_title('Fit parameters of MS lifetime')
            ax[len(params)-2].set_xlabel(r'$\mathrm{[Fe/H]}$')

            if stat:
                ax[0].text(0.05, 0.8, 
                        r'$\mathrm{<\chi^2> \, = \, }$'+ str(chi2_mean) + r'$\, \pm \,$' + str(chi2_std), 
                        transform=ax[0].transAxes
                        )
            
            for i, key in enumerate(labels[1:]):
                    
                ax[i].plot(FeH_grid, output[key], 'o-',c=lbs[key]['c'])
                ax[i].set_ylabel(lbs[key]['l'])
                if i < len(params)-2:
                    ax[i].set_xticklabels([])

            if func != 'v1':
                figname = 'MS_lifetime_fit_params' + '_' + func + '.png'
            else:
                figname = 'MS_lifetime_fit_params' + '_' + func + '_Mbr' + str(Mbr) + '.png'
            plt.savefig(os.path.join(dir_out,figname))

            msl1['fig'] = (f, ax)
            
        return output, msl1

