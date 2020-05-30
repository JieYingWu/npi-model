import sys
import numpy as np
import os
from os.path import join, exists
import pandas as pd
from data_parser import get_data, Processing
from statsmodels.distributions.empirical_distribution import ECDF
from utils import *


class CountyGenerator():

    def __init__(self, N2, si, num_alphas, alpha_mu, alpha_var):
        super(CountyGenerator, self).__init__()
        self.N2 = N2
        self.si = si
        self.alpha_mu = alpha_mu
        self.alpha_var = alpha_var
        self.generate_alphas(num_alphas)

        wf_file = join('data', 'us_data', 'weighted_fatality_new.csv')
        self.weighted_fatalities = pd.read_csv(wf_file, encoding='latin1', index_col='FIPS')


    # Generate all alphas for this object (cluster)
    def generate_alphas(self, num_alphas):
        shape = self.alpha_var**-2
        scale = shape / self.alpha_mu
        alphas = np.random.gamma(self.alpha_mu, self.alpha_var, num_alphas)/2.5
#        alphas = np.ones(num_alphas)*0.2
        self.alphas = -1*alphas
        print(self.alphas)

        
    # Generate fataility rates or read from cached 
    def calculate_fatality_rate(self, region):
            
        ifr = self.weighted_fatalities.loc[int(region), 'fatality_rate']
        SI = self.si[0:self.N2]

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
    
        all_f = np.zeros((self.N2))
        ifr = float(ifr)
    
        ## assume that IFR is probability of dying given infection
        x1 = np.random.gamma(alpha1, beta1, 5000000) # infection-to-onset -> do all people who are infected get to onset?
        x2 = np.random.gamma(alpha2, beta2, 5000000) # onset-to-death
        f = ECDF(x1+x2)
        def conv(u): # IFR is the country's probability of death
            return ifr * f(u)

        h = np.zeros(self.N2) # Discrete hazard rate from time t = 1, ..., 100
        h[0] = (conv(1.5) - conv(0.0))

        for i in range(1, self.N2):
            h[i] = (conv(i+.5) - conv(i-.5)) / (1-conv(i-.5))
        s = np.zeros(self.N2)
        s[0] = 1
        for i in range(1, self.N2):
            s[i] = s[i-1]*(1-h[i-1])
                
        return s * h

        
    # Generate a sequence of Rt for a county given the R0 and a sequence of intervention dates and weights
    def calculate_rt(self, r0, interventions):
        effects = np.sum(interventions * self.alphas, axis=1)
        factor = np.exp(effects)
        rt = r0*factor
        return rt

    
    # Generate the number of cases given some Rt
    def predict_cases(self, rt):
#        tau = np.random.exponential(0.03, (6)) # Seed the first 6 days
        si = self.si[::-1]
        prediction = np.zeros(rt.shape[0])
        prediction[0:6] = 100 #np.exp(np.arange(6))*6
#        print(prediction)
#        exit()
        for i in range(6, rt.shape[0]):
            prediction[i] = rt[i] * np.sum(prediction[0:i] * si[-i:])

        return prediction.astype(np.int)

    
    # Generate the number of deaths given some prediction
    def predict_deaths(self, rt, prediction, fatality):
        f = fatality[::-1]
        deaths = np.zeros(rt.shape[0])
        for i in range(1,rt.shape[0]):
            deaths[i] = np.sum(prediction[0:i] * f[-i:])

        return deaths.astype(np.int)

    
    # Create the rt, cases, and deaths timeseries given a region characteristics
    def make_county(self, r0, interventions, region):
        rt = self.calculate_rt(r0, interventions)
        cases = self.predict_cases(rt)
        fatality = self.calculate_fatality_rate(region)
        deaths = self.predict_deaths(rt, cases, fatality)

        for i in range(1, cases.shape[0]):
            cases[i] = cases[i] + cases[i-1]
            deaths[i] = deaths[i] + deaths[i-1]
        return rt, cases, deaths

    
# Get interventions as binary timeseries
def parse_interventions(stan_data, data_dir='data'):
    i1 = np.expand_dims(stan_data['covariate1'], axis=2)
    i2 = np.expand_dims(stan_data['covariate2'], axis=2)
    i3 = np.expand_dims(stan_data['covariate3'], axis=2)
    i4 = np.expand_dims(stan_data['covariate4'], axis=2)
    i5 = np.expand_dims(stan_data['covariate5'], axis=2)
    i6 = np.expand_dims(stan_data['covariate6'], axis=2)
    i7 = np.expand_dims(stan_data['covariate7'], axis=2)
    i8 = np.expand_dims(stan_data['covariate8'], axis=2)
    interventions = np.concatenate((i1, i2, i3, i4, i5, i6, i7, i8), axis=2)
    interventions = interventions.transpose(1, 0, 2)
    return interventions
    
    
if __name__ == '__main__':
    data_dir = sys.argv[1]
    N2 = 120

    alpha_mu = 0.5
    alpha_var = 1
    num_alphas = 8

#    interventions = get_npis(data_dir)
#    regions = get_counties_isolated_NPIs(interventions, 'public schools').values.tolist()
#    regions.sort()

#    for i in range(len(regions)):
#        regions[i] = str(regions[i])
    stan_data, regions, start_date, geocode = get_data(100, 'data', processing=Processing.REMOVE_NEGATIVE_VALUES, state=False)

    r0_file_path = join('results', 'real_county', 'summary.csv')
    r0_file = pd.read_csv(r0_file_path)
    
    means = r0_file['mean'].values
    M = len(geocode)
    means= means[0:M]

    all_r0 = {}
    for r in range(M):
        all_r0[str(geocode[r]).zfill(5)] = means[r]

    serial_interval = np.loadtxt(join('data', 'us_data', 'serial_interval.csv'), skiprows=1, delimiter=',')
    si = serial_interval[:,1]

    generator = CountyGenerator(N2, si, num_alphas, alpha_mu, alpha_var)
#    generator.alphas = [-0.124371438107218, -0.196069499889346, -0.194197939254073, -0.495431571118872, -0.378146551081655, -0.137932933788039, -0.29558366952368, -0.422007707986038]

    interventions = parse_interventions(stan_data)
    all_rt = {}
    all_cases = {}
    all_deaths = {}

    for r in range(M):
        region = geocode[r]
        r0 = all_r0[region]
        intervention = interventions[r,:,:]
        rt, cases, deaths = generator.make_county(r0, intervention, region)
        all_rt[region] = rt
        all_cases[region] = cases
        all_deaths[region] = deaths


    dtype = dict(FIPS=str)
    summary_path = join(data_dir, 'us_data', 'summary.csv')
    interventions_path = join(data_dir, 'us_data', 'interventions_timeseries.csv')
    cases_path = join(data_dir, 'us_data', 'infections_timeseries_w_states.csv')
    deaths_path = join(data_dir, 'us_data', 'deaths_timeseries_w_states.csv')

    real_cases_path = join('data', 'us_data', 'infections_timeseries_w_states.csv')
    real_deaths_path = join('data', 'us_data', 'deaths_timeseries_w_states.csv')
    real_cases_df = pd.read_csv(real_cases_path, dtype=dtype)
    real_deaths_df = pd.read_csv(real_deaths_path, dtype=dtype)
    real_cases_df = real_cases_df.set_index('FIPS')
    real_deaths_df = real_deaths_df.set_index('FIPS')

    
    summary = {'N2':N2, 'alpha_mu':alpha_mu, 'alpha_var':alpha_var, 'alphas':generator.alphas}
    df = pd.DataFrame.from_dict(summary)
    df.to_csv(summary_path)
    
    cases_df = real_cases_df.copy()
    deaths_df = real_deaths_df.copy()

    for r in range(M):
        region = str(geocode[r]).zfill(5)

        simulated_cases = all_cases[region][0:len(real_cases_df.loc[region, start_date[r]:])]
        cases_df.loc[region, start_date[r]:] = simulated_cases
        
        simulated_deaths = all_deaths[region][0:len(real_deaths_df.loc[region, start_date[r]:])]
        deaths_df.loc[region, start_date[r]:] = simulated_deaths

    cases_df.to_csv(cases_path)
    deaths_df.to_csv(deaths_path)
    rt_df = pd.DataFrame.from_dict(all_rt)
    rt_df.to_csv(interventions_path)
