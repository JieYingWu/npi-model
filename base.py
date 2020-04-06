from os.path import join, exists
import sys
import numpy as np
from data_parser import get_stan_parameters, get_stan_parameters_our
import pystan
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import pickle
import datetime
from forecast_plots import plot_forecasts

assert len(sys.argv) == 3

# Compile the model
sm = pystan.StanModel(file='stan-models/base.stan')

data_dir = sys.argv[1]
weighted_fatalities = np.loadtxt(join(data_dir, 'weighted_fatality.csv'), skiprows=1, delimiter=',', dtype=str)
ifrs = {}


if sys.argv[2] == 'europe':
    stan_data, plot_data, countries = get_stan_parameters(data_dir)
    for i in range(weighted_fatalities.shape[0]):
        ifrs[weighted_fatalities[i,1]] = float(weighted_fatalities[i,-2])

elif sys.argv[2] == 'US':
    stan_data, plot_data, countries = get_stan_parameters_our(20, data_dir)
    for i in range(weighted_fatalities.shape[0]):
        ifrs[str(weighted_fatalities[i,0])] = weighted_fatalities[i,-1]


N2 = stan_data['N2']
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

all_f = np.zeros((N2, len(countries)))
for c in range(len(countries)):
    ifr = float(ifrs[str(countries[c])])
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

    all_f[:,c] = s * h

stan_data['f'] = all_f

## TODO: fill in the data for the stan model - check if Python wants something different
#stan_data = list(M=length(countries),N=NULL,p=p,x1=poly(1:N2,2)[,1],x2=poly(1:N2,2)[,2],
#                 y=NULL,covariate1=NULL,covariate2=NULL,covariate3=NULL,covariate4=NULL,covariate5=NULL,covariate6=NULL,covariate7=NULL,deaths=NULL,f=NULL,
#                 N0=6,cases=NULL,LENGTHSCALE=7,SI=serial.interval$fit[1:N2],
#                 EpidemicStart = NULL) # N0 = 6 to make it consistent with Rayleigh
#stan_data = {'M':len(countries), 'N':N, 'p':interventions.shape[1]-1,...}

# Train the model and generate samples - returns a StanFit4Model
fit = sm.sampling(data=stan_data, iter=200, chains=4, warmup=100, thin=4, seed=101, control={'adapt_delta':0.9, 'max_treedepth':10})
# fit = sm.sampling(data=stan_data, iter=20, chains=4, warmup=10, thin=4, seed=101, control={'adapt_delta':0.9, 'max_treedepth':10})

# All the parameters in the stan model
# mu = fit['mu']
# alpha = fit['alpha']
# kappa = fit['kappa']
# y = fit['y']
# phi = fit['phi']
# tau = fit['tau']
# prediction = fit['prediction']
# estimated_deaths = fit['E_deaths']
# estimated_deaths_cf = fit['E_deaths0']
# print(mu, alpha, kappa, y, phi, tau, prediction, estimated_deaths, estimated_deaths_cf)

summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'], 
                 columns=summary_dict['summary_colnames'], 
                 index=summary_dict['summary_rownames'])
df.to_csv(r'summary.csv', sep=';')

## TODO: Make pretty plots
## use plot_data to get start_dates and geocode data for plotting
# Probably don't have to use Imperial data for this, just find similar looking Python packages
# data_country = pd.DataFrame({'time': s, 'deaths': })
# plot_forecasts()
