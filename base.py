from os.path import join, exists
import sys
import numpy as np

import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## Copied from https://towardsdatascience.com/an-introduction-to-bayesian-inference-in-pystan-c27078e58d53
# sns.set()  # Nice plot aesthetic
# np.random.seed(101)

# model = """

# data {
#     int<lower=0> N;
#     vector[N] x;
#     vector[N] y;
# }
# parameters {
#     real alpha;
#     real beta;
#     real<lower=0> sigma;
# }
# model {
#     y ~ normal(alpha + beta * x, sigma);
# }

# """

# # Parameters to be inferred
# alpha = 4.0
# beta = 0.5
# sigma = 1.0

# # Generate and plot data
# x = 10 * np.random.rand(100)
# y = alpha + beta * x
# y = np.random.normal(y, scale=sigma)

# # Put our data in a dictionary
# #data = {'N': len(x), 'x': x, 'y': y}

# Compile the model
sm = pystan.StanModel(file='stan-models/base.stan')


data_dir = sys.argv[1]

countries = ['Denmark', 'Italy', 'Germany', 'Spain', 'United Kingdom', 'France', 'Norway', 'Belgium', 'Austria', 'Sweden', 'Switzerland']
serial.interval = read.csv(join(data_dir, 'serial_interval.csv')) # Time between primary infector showing symptoms and secondary infected showing symptoms

interventions = np.loadtxt(join(data_dir, 'interventions.csv'))
## TODO: They check that if any measure has not been in place until lockdown, set that intervention date to lockdown
## TODO: Build an array of intervention days where rows are the countries and columns are the interventions in this order: school/uni closures, self-isolating if ill, bannig public events, any government intervention, complete/partial lockdown, and social distancing/isolation

weighted_fatalities = np.loadtxt(join(data_dir, 'weighted_fatality.csv'))
## TODO: I think we just need the weight fatailites column to calculate probability of death

covid_up_to_date = np.loadtxt(join(data_dir, 'COVID-19-up-to-date.csv'))
## TODO: covid_up_to_date is in a horrendous format - might consider reshaping for later use
## TODO: Extract start date of relevant countries

## TODO: Build infections/deaths timeseries for relevant countries
# cases = 
# deaths = 

N = shape(cases)[1]
N2 = 75


## TODO: turn rgammAlt, ecdf, and function thing into Python gamma distribution and convolution
# infection to onset
mean1 = 5.1
cv1 = 0.86
# onset to death
mean2 = 18.8; cv2 = 0.45 
## assume that IFR is probability of dying given infection
x1 = rgammaAlt(5e6,mean1,cv1) # infection-to-onset -> do all people who are infected get to onset?
x2 = rgammaAlt(5e6,mean2,cv2) # onset-to-death
f = ecdf(x1+x2)
convolution = function(u) (IFR * f(u)) # IFR is the country's probability of death

## TODO: fill in the data for the stan model - check if Python wants something different
stan_data = list(M=length(countries),N=NULL,p=p,x1=poly(1:N2,2)[,1],x2=poly(1:N2,2)[,2],
                 y=NULL,covariate1=NULL,covariate2=NULL,covariate3=NULL,covariate4=NULL,covariate5=NULL,covariate6=NULL,covariate7=NULL,deaths=NULL,f=NULL,
                 N0=6,cases=NULL,LENGTHSCALE=7,SI=serial.interval$fit[1:N2],
                 EpidemicStart = NULL) # N0 = 6 to make it consistent with Rayleigh
#stan_data = [len(countries), N=N, p=interventions.shape[1]-1,...]

# Train the model and generate samples
fit = sm.sampling(data=data, iter=200, chains=4, warmup=100, thin=4, seed=101, control={adapt_delta:0.9, max_treedepth:10})

## TODO: Read out the data of the stan model



## TODO: Make pretty plots
