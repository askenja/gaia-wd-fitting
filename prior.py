'''
Here mean and standard deviation of the Gaussian priors for the parameters are defined.
Not all of these parameters are optimized in practice, 
different MCMC runc can be performed with different sets of free parameters. 

E.g., thick-disk SFR parameters (tt1, gamma, beta), 
thin-disk smaller SFR peak parameters (sigmap1, tpk1), cannot be constrained 
with the selected data. 
'''


prior = {
    'm_br1': {
        'm': 2.8,  # IFMR Cummings+18 (C18)
        's': 0.14  # max possible value to always have m_br2 > m_br1
        },   
    'm_br2': {
        'm': 3.65, # IFMR C18
        's': 0.14  # max possible value to always have m_br2 > m_br1
        }, 
    'alpha1': {
        'm': 0.12, # modified value, IFMR C18 = 0.0873
        's': 0.02   # C18 has 0.019; I tested 0.025 = almost max possible, 0.029, to keep alpha1 > 0, but maybe it's too wide prior, so reduced to 0.02
        }, 
    'alpha2': {
        'm': 0.05, # modified value, IFMR C18 = 0.181
        's': 0.04   # 0.041 in C18; I tested 0.06 = approx. max possible to have alpha2 > 0 (no inversion of the slope sign), but now reduced to 0.04
        }, 
    'alpha3': {
        'm': 0.1, # modified value, IFMR C18 = 0.0835
        's': 0.015 # 0.0144 in C18; I tested 0.025 = approx. max possible to have alpha3 > 0, now reduced to 0.015
        }, 
    'alpha_cool': {
        'm': 0.0, # Default is NO cooling slowdown
        's': 0.1  # Dispersion of the slowdown of cooling assumed to be 10%
        }, 
    'td2': {
        'm': 7.8, # Sysoliatina and Just (2021), SJ21
        's': 2.59  # allows td2 cloze to 0 -> approx. const SFR can be tested! 
        },
    'dzeta': {
        'm': 0.83,  # SJ21
        's': 0.27   # As wide as possible to keep value positive
        },
    'eta': {
        'm': 5.6, # SJ21
        's': 1.5  # allows large enough variation of SFR shape in combination with dzeta
        },
    'sigmad': {
        'm': 29.4, # SJ21, total thin-disk SFR normalization
        's': 3.0   # wide enough prior, but only minimal change expected
        },
    'sigmap0': {
        'm': 3.5,  # SJ21, normalization of SFR secondary peak at ~3 Gyr ago
        's': 1.15  # allows going back almost to 0 
        },
    'sigmap1': {
        'm': 1.3,  # SJ21, normalization of SFR secondary peak at ~0.5 Gyr ago
        's': 0.4  # allows going back almost to 0 
        },
    'tpk0': {
        'm': 10.0, # SJ21, time (not age! 10 -> 13-10=3 Gyr ago) of the first secondary thin-disk SFR peak
        's': 0.99  # range up to 13 Gyr, max model age; 0.99 instead of 1 to avoid ages == 13 Gyr, causes an error
        },
    'tpk1': {
        'm': 12.5, # SJ21, time of the second secondary thin-disk SFR peak
        's': 0.15  # range up to 13 Gyr, max model age; 0.99 instead of 1 to avoid ages == 13 Gyr, causes an error
        },
    'tt1': {
        'm': 0.1,  # SJ21, thick-disk SFR parameter
        's': 0.03  # range almost from 0 up to a const
        },
    'gamma': {
        'm': 2.0,  # SJ21, thick-disk SFR parameter
        's': 0.6   # to have range almost from 0 + large shape variation
        },
    'beta': {
        'm': 3.5,  # SJ21, thick-disk SFR parameter
        's': 1.0   # - // -
        },
    'a0': {
        'm': 1.31, # SJ21, IFM slope for 0.08 Msun < m_ini < m0
        's': 0.15  # as in SJ21, large enough to have significant slope variation
        }, 
    'a1': {
        'm': 1.5,  # SJ21, IFM slope for m0 < m_ini < m1
        's': 0.15  # - // -
        }, 
    'a2': {
        'm': 2.88, # SJ21, IFM slope for m1 < m_ini < 6 Msun
        's': 0.15  # - // -
        }, 
    'm0': {
        'm': 0.49, # SJ21, IFM break point
        's': 0.15  # max possible value to have m1 > m0
        }, 
    'm1': {
        'm': 1.43, # SJ21, IFM break point
        's': 0.15  # - // -
        },
    'f_da': {
        'm': 0.8,  # As often in the literature, fraction of DA WDs
        's': 0.065 # max value to have f_da < 1 given mean value 0.8. 
        }
}