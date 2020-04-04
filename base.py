from os.path import join, exists
import sys
import numpy as np
from data_parser import get_stan_parameters, get_stan_parameters_our

import pystan
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

# Compile the model
sm = pystan.StanModel(file='stan-models/base.stan')

data_dir = sys.argv[1]

countries = ['Denmark', 'Italy', 'Germany', 'Spain', 'United Kingdom', 'France', 'Norway', 'Belgium', 'Austria', 'Sweden', 'Switzerland']
serial_interval = np.loadtxt(join(data_dir, 'serial_interval.csv'), skiprows=1, delimiter=',') # Time between primary infector showing symptoms and secondary infected showing symptoms - this is a probability distribution from 1 to 100 days


#interventions = np.loadtxt(join(data_dir, 'interventions.csv'))
## TODO: They check that if any measure has not been in place until lockdown, set that intervention date to lockdown
## TODO: Build an array of intervention days where rows are the countries and columns are the interventions in this order:
# school/uni closures, self-isolating if ill, bannig public events, any government intervention, complete/partial lockdown, and social distancing/isolation

weighted_fatalities = np.loadtxt(join(data_dir, 'weighted_fatality.csv'), skiprows=1, delimiter=',', dtype=str)
ifrs = {}

for i in range(weighted_fatalities.shape[0]):
    ifrs[weighted_fatalities[i,1]] = float(weighted_fatalities[i,-2])

## TODO: I think we just need the weight fatailites column to calculate probability of death

stan_data = get_stan_parameters()
N2 = stan_data['N2']
print(serial_interval.shape)
print(stan_data['N2'])
SI = serial_interval[0:stan_data['N2'],1]
stan_data['SI'] = SI

print(stan_data)

## TODO: turn rgammAlt, ecdf, and function thing into Python gamma distribution and convolution
# infection to onset
mean1 = 5.1
cv1 = 0.86
# onset to death
mean2 = 18.8
cv2 = 0.45 

all_f = np.zeros((N2, len(countries)))
for c in range(len(countries)):
    ifr = ifrs[countries[c]]
    
    h = np.zeros(N2) # Discrete hazard rate from time t = 1, ..., 100

    ## assume that IFR is probability of dying given infection
    x1 = np.random.gamma(mean1, cv1, 5000000) # infection-to-onset -> do all people who are infected get to onset?
    x2 = np.random.gamma(mean2, cv2, 5000000) # onset-to-death

    f = ECDF(x1+x2)

    def conv(u): # IFR is the country's probability of death
        return ifr * f(u)
    
    h[0] = (conv(1.5) - conv(0))
    
    for i in range(1, N2):
        h[i] = (conv(i+.5) - conv(i-.5)) / (1-conv(i-.5))

    s = np.zeros(N2)
    s[0] = 1
    for i in range(1, N2):
        s[i] = s[i-1]*(1-h[i-1])
    all_f[:,c] = s * h

stan_data['f'] = all_f
print(stan_data.keys())
## TODO: fill in the data for the stan model - check if Python wants something different
#stan_data = list(M=length(countries),N=NULL,p=p,x1=poly(1:N2,2)[,1],x2=poly(1:N2,2)[,2],
#                 y=NULL,covariate1=NULL,covariate2=NULL,covariate3=NULL,covariate4=NULL,covariate5=NULL,covariate6=NULL,covariate7=NULL,deaths=NULL,f=NULL,
#                 N0=6,cases=NULL,LENGTHSCALE=7,SI=serial.interval$fit[1:N2],
#                 EpidemicStart = NULL) # N0 = 6 to make it consistent with Rayleigh
#stan_data = {'M':len(countries), 'N':N, 'p':interventions.shape[1]-1,...}

# Train the model and generate samples - returns a StanFit4Model
fit = sm.sampling(data=stan_data, iter=200, chains=4, warmup=200, thin=4, seed=101, control={'adapt_delta':0.9, 'max_treedepth':10})

## TODO: Read out the data of the stan model
# Seems like extract parameters by str of the parameter name: https://pystan.readthedocs.io/en/latest/api.html#stanfit4model
# Check that Rhat is close to 1 to see if the model's converged

summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'], 
                  columns=summary_dict['summary_colnames'], 
                  index=summary_dict['summary_rownames'])

#alpha_mean, beta_mean = df['mean']['alpha'], df['mean']['beta']

# All the parameters in the stan model
mu = fit['mu']
alpha = fit['alpha']
kappa = fit['kappa']
y = fit['y']
phi = fit['phi']
tau = fit['tau']
prediction = fit['prediction']
estimated_deaths = fit['E_deaths']
estimated_deaths_cf = fit['E_deaths0']

print(mu, alpha, kappa, y, phi, tau, prediction, estimated_deaths, estimated_deaths_cf)
## TODO: Make pretty plots
# Probably don't have to use Imperial data for this, just find similar looking Python packages
