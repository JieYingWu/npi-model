from os.path import join
import sys
import numpy as np
from data_parser import get_data, Processing
from data_parser_europe import get_data_europe
import pystan
import datetime as dt
from dateutil.parser import parse
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

regions = [55079, 53033, 42101, 36119, 36103, 36087, 36071, 36061, 36059, 36055, 36029, 34039, 34035, 34031, 34029, 34027, 34025, 34023, 34021, 34017, 34013, 34007, 34005, 34003, 32003, 29189, 27053, 26163, 26125, 26099, 26049, 24033, 24031, 24005, 24003, 22103, 22071, 22051, 22033, 18097, 18089, 17197, 17097, 17043, 17031, 12099, 12086, 12011, 11001, 9009, 9007, 9003, 9001, 6073, 6065, 6037, ]
regions = regions[::-1]
M = len(regions) # 56 I think

data_dir = 'data'
stan_data, regions, start_date, geocode = get_data(M, data_dir, processing=Processing.REMOVE_NEGATIVE_VALUES, state=False, fips_list=regions)
wf_file = join(data_dir, 'us_data', 'weighted_fatality.csv')
weighted_fatalities = np.loadtxt(wf_file, skiprows=1, delimiter=',', dtype=str)


N2 = stan_data['N2']

# Build a dictionary of region identifier to weighted fatality rate
ifrs = {}
for i in range(weighted_fatalities.shape[0]):
    ifrs[weighted_fatalities[i,0]] = weighted_fatalities[i,-1]
stan_data['cases'] = stan_data['cases'].astype(np.int)
stan_data['deaths'] = stan_data['deaths'].astype(np.int)

sm = pystan.StanModel(file='stan-models/us_new.stan')


serial_interval = np.loadtxt(join(data_dir, 'serial_interval.csv'), skiprows=1, delimiter=',')
# Time between primary infector showing symptoms and secondary infected showing symptoms - this is a probability distribution from 1 to 100 days

SI = serial_interval[0:stan_data['N2'],1]
stan_data['SI'] = SI

# infection to onset
mean1 = 5.1
cv1 = 0.86
alpha1 = cv1**-2
beta1 = mean1/alpha1
# onset to death
mean2 = 18.8
cv2 = 0.45
alpha2 = cv2**-2
beta2 = mean2/alpha2

all_f = np.zeros((N2, len(regions)))
for r in range(len(regions)):
    ifr = float(ifrs[str(regions[r])])
    
    ## assume that IFR is probability of dying given infection
    x1 = np.random.gamma(alpha1, beta1, 5000000) # infection-to-onset -> do all people who are infected get to onset?
    x2 = np.random.gamma(alpha2, beta2, 5000000) # onset-to-death
    f = ECDF(x1+x2)
    def conv(u): # IFR is the country's probability of death
        return ifr * f(u)

    h = np.zeros(N2) # Discrete hazard rate from time t = 1, ..., 100
    h[0] = (conv(1.5) - conv(0.0))

    for i in range(1, N2):
        h[i] = (conv(i+.5) - conv(i-.5)) / (1-conv(i-.5))
    s = np.zeros(N2)
    s[0] = 1
    for i in range(1, N2):
        s[i] = s[i-1]*(1-h[i-1])

    all_f[:,r] = s * h

stan_data['f'] = all_f

    
fit = sm.sampling(data=stan_data, iter=200, chains=4, warmup=100, thin=4, control={'adapt_delta':0.9, 'max_treedepth':10})
# fit = sm.sampling(data=stan_data, iter=2000, chains=4, warmup=10, thin=4, seed=101, control={'adapt_delta':0.9, 'max_treedepth':10})

summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'],
                 columns=summary_dict['summary_colnames'],
                 index=summary_dict['summary_rownames'])

region = 'US_county'
df.to_csv('results/' + region + '_summary.csv', sep=',')

df_sd = pd.DataFrame(start_date, index=[0])
df_geo = pd.DataFrame(geocode, index=[0])
df_sd.to_csv('results/' + region + '_start_dates.csv', sep=',')
df_geo.to_csv('results/' + region + '_geocode.csv', sep=',')
