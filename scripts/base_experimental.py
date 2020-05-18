import os
from os.path import join, exists
import sys
import numpy as np
from data_parser import get_data, Processing
from data_parser_europe import get_data_europe
import pystan
import datetime as dt
from dateutil.parser import parse
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from utils import get_cluster
import pickle

def main():
    #    regions = [55079, 53033, 42101, 36119, 36103, 36087, 36071, 36061, 36059, 36055, 36029, 34039, 34035, 34031, 34029, 34027, 34025, 34023, 34021, 34017, 34013, 34007, 34005, 34003, 32003, 29189, 27053, 26163, 26125, 26099, 26049, 24510, 24033, 24031, 24005, 24003, 22103, 22071, 22051, 22033, 18097, 18089, 17197, 17097, 17043, 17031, 13121, 12099, 12086, 12011, 11001, 9009, 9007, 9003, 9001, 6073, 6065, 6037, ]
    #regions = regions[::-1]
    regions = [1073, 8115, 13095, 15007, 18051, 29095, 37049, 48041, 48157, 48439,
               55079, 53033, 42101, 47089, 36119, 50027, 20101, 29077, 34025, 18097]
    
    # regions = [31079, 36087, 35045, 48419, 6107, 1073, 4017, 35031, 22097, 39095, 28035, 18089, 22045, 48479, 22039, 22047, 35029, 13299, 22057, 5119,
    #            13035, 22037, 13141, 1005, 40107, 39101, 39121, 39141, 40003, 40005, 40035, 40055, 40063, 40151, 37177, 42053, 45005, 45037, 45049, 45061,
    #            34029, 12099, 12081, 31081, 10005, 4015, 12103, 25001, 37089, 12021, 40041, 12071, 24029, 26007, 12085, 25003, 12101, 12069, 26001, 12127,
    #            17031, 34003, 34031, 6037, 26163, 34013, 25017, 34017, 34023, 34039, 9001, 33011, 26125, 9003, 34027, 36119, 34025, 25009, 18097, 42091,
    #            39093, 33013, 37047, 1017, 21085, 17027, 37037, 26155, 28075, 28057, 9011, 9005, 48477, 28069, 28085, 28109, 5075, 5069, 51065, 40097]
     
    #filename = 'data/us_data/clustering.csv'
    #    tag = 'simulated_county' # 
    #    tag = 'real_county' 
    regions.sort()
    M = len(regions)
    print('Running for ' + str(M) + ' FIPS')
    
    data_dir = sys.argv[1]
    tag = sys.argv[2]
    stan_data, regions, start_date, geocode = get_data(M, data_dir, processing=Processing.REMOVE_NEGATIVE_VALUES, state=False, fips_list=regions)
    #print(stan_data['M'])
    #exit()
    wf_file = join(data_dir, 'us_data', 'weighted_fatality_new.csv')
    weighted_fatalities = np.loadtxt(wf_file, skiprows=1, delimiter=',', dtype=str)
    
    #    print(stan_data['M'])
    N2 = stan_data['N2']
    
    
    wf_file = join(data_dir, 'us_data', 'weighted_fatality_new.csv')
    ifrs = pd.read_csv(wf_file, encoding='latin1', index_col='FIPS')
    
    stan_data['cases'] = stan_data['cases'].astype(np.int)
    stan_data['deaths'] = stan_data['deaths'].astype(np.int)
    
    sm = pystan.StanModel(file='stan-models/us_new.stan')
    
    serial_interval = np.loadtxt(join(data_dir, 'serial_interval.csv'), skiprows=1, delimiter=',')
    # Time between primary infector showing symptoms and secondary infected showing symptoms - this is a probability distribution from 1 to 100 days

    n_si = len(serial_interval)
    SI = np.zeros(stan_data['N2'])
    SI[0:n_si] = serial_interval[0:n_si,1]
    if n_si < stan_data['N2']:
        SI[n_si : stan_data['N2']] = 0.0
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
        ifr = ifrs.loc[regions[r], 'fatality_rate']
        
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

    print(stan_data['M'])
    
    path = join('results', tag)
    if not exists(path):
        os.makedirs(path)
    
    stan_file_path = join(path, 'stan_file.pkl')
    with open(stan_file_path, 'wb') as f:
        pickle.dump(stan_data, f)
    
    print ("Start sampling -------------------")
    
    fit = sm.sampling(data=stan_data, iter=500, chains=4, warmup=250, thin=4, n_jobs=8, control={'adapt_delta':0.95, 'max_treedepth':15})
    # fit = sm.sampling(data=stan_data, iter=1000, chains=4, warmup=500, thin=4, control={'adapt_delta':0.9, 'max_treedepth':12})
    # fit = sm.sampling(data=stan_data, iter=2000, chains=4, warmup=10, thin=4, seed=101, control={'adapt_delta':0.9, 'max_treedepth':10})
    print ("Fitted model --------------------")
    summary_dict = fit.summary()
    df = pd.DataFrame(summary_dict['summary'],
                     columns=summary_dict['summary_colnames'],
                     index=summary_dict['summary_rownames'])
    
    df.to_csv(join(path, 'summary.csv'), sep=',')
    
    df_sd = pd.DataFrame(start_date, index=[0])
    df_geo = pd.DataFrame(geocode, index=[0])
    df_sd.to_csv(join(path, 'start_dates.csv'), sep=',')
    df_geo.to_csv(join(path, 'geocode.csv'), sep=',')

if __name__ == '__main__':
    main()
