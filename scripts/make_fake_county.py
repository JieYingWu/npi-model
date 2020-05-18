import numpy as np
import os
from os.path import join, exists
import pandas as pd
from data_parser import get_data, Processing
from statsmodels.distributions.empirical_distribution import ECDF
import utils

class CountyGenerator():

    def __init__(self, N2, si, num_alphas, alpha_mu, alpha_var, type_of_alpha):
        super(CountyGenerator, self).__init__()
        self.N2 = N2
        self.si = si
        self.alpha_mu = alpha_mu
        self.alpha_var = alpha_var
        self.generate_alphas(num_alphas, type_of_alpha)

        wf_file = join(data_dir, 'us_data', 'weighted_fatality_new.csv')
        self.weighted_fatalities = pd.read_csv(wf_file, encoding='latin1', index_col='FIPS')


    # Generate all alphas for this object (cluster)
    def generate_alphas(self, num_alphas, type_of_alpha):
        if type_of_alpha == 'random':
            alphas = np.random.normal(self.alpha_mu, self.alpha_var, num_alphas)
            self.alphas = -1*alphas
        elif type_of_alpha=='same':
            self.alphas = [0.852450511,
                            0.173904206,
                            0.100636835,
                            0.090076732,
                            0.060130794,
                            0.050961405,
                            0.058717003,
                            0.030499885
                            ]

        
    # Generate fataility rates or read from cached 
    def calculate_fatality_rate(self, region):
            
        ifr = self.weighted_fatalities.loc[region, 'fatality_rate']
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
        tau = np.random.exponential(0.03) # Seed the first 6 days
        si = self.si[::-1]
        prediction = np.zeros(rt.shape[0])
        prediction[0:6] = np.random.exponential(1/tau)
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
def parse_interventions(regions, data_dir='data'):
    stan_data, regions, start_date, geocode = get_data(len(regions), data_dir, processing=Processing.REMOVE_NEGATIVE_VALUES, state=False, fips_list=regions)
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
    return interventions, start_date, geocode
    
    
if __name__ == '__main__':
    data_dir = 'simulated'
    N2 = 100

    r0_file_path = join('results', 'real_county', 'summary.csv')
    r0_file = pd.read_csv(r0_file_path)
    geocode_path = join('results', 'real_county', 'geocode.csv')
    geocode_file = pd.read_csv(geocode_path)
    geocode = geocode_file.values[0][1:]
    interventions_path = join('data', 'us_data/interventions.csv')
    interventions = pd.read_csv(interventions_path)


    alpha_mu = 0.4
    alpha_var = 0.2
    num_alphas = 8
    
    type_of_alpha = 'same'

#    regions = [55079, 53033, 42101, 36119, 36103, 36087, 36071, 36061, 36059, 36055, 36029, 34039, 34035, 34031, 34029, 34027, 34025, 34023, 34021, 34017, 34013, 34007, 34005, 34003, 32003, 29189, 27053, 26163, 26125, 26099, 26049, 24510, 24033, 24031, 24005, 24003, 22103, 22071, 22051, 22033, 18097, 18089, 17197, 17097, 17043, 17031, 13121, 12099, 12086, 12011, 11001, 9009, 9007, 9003, 9001, 6073, 6065, 6037]
#    regions = [1073, 8000, 8001, 8003, 8005, 8007, 8009, 8011, 8013, 8014, 8015, 8017, 8019, 8021, 8023, 8025, 8027, 8029, 8031, 8033, 8035, 8037, 8039, 8041, 8043, 8045, 8047, 8049, 8051, 8053, 8055, 8057, 8059, 8061, 8063, 8065, 8067, 8069, 8071, 8073, 8075, 8077, 8079, 8081, 8083, 8085, 8087, 8089, 8091, 8093, 8095, 8097, 8099, 8101, 8103, 8105, 8107, 8109, 8111, 8113, 8115, 8117, 8119, 8121, 8123, 8125, 48015, 48027, 48029, 48041, 48085, 48141, 48157, 48167, 48201, 48309, 48439]
    regions = [1073, 8115, 13095, 15007, 18051, 29095, 37049, 48041, 48157, 48439,
               55079, 53033, 42101, 47089, 36119, 50027, 20101, 29077, 34025, 18097]
    regions.sort()
    
    n = len(regions)
    means = r0_file['mean'].values
    means= means[0:n]
    all_r0 = {}
    for i in range(n):
        all_r0[geocode[i]] = means[i]
    
    serial_interval = np.loadtxt(join(data_dir, 'serial_interval.csv'), skiprows=1, delimiter=',')
    si = serial_interval[:,1]

    generator = CountyGenerator(N2, si, num_alphas, alpha_mu, alpha_var, type_of_alpha)
#    generator.alphas = [-0.124371438107218, -0.196069499889346, -0.194197939254073, -0.495431571118872, -0.378146551081655, -0.137932933788039, -0.29558366952368, -0.422007707986038]

    interventions, start_date, geocode_intervention = parse_interventions(regions)
    all_rt = {}
    all_cases = {}
    all_deaths = {}
    
    for r in range(len(regions)):
        region = geocode[r]
        print(region)
        break
        r0 = all_r0[region]
        intervention = interventions[r,:,:]
        rt, cases, deaths = generator.make_county(r0, intervention, region)
        all_rt[region] = rt
        all_cases[region] = cases
        all_deaths[region] = deaths

    name_summary = 'summary_' + type_of_alpha + '.csv'
    summary_path = join(data_dir, 'us_data', name_summary)
    name_interventions = 'interventions_timeseries_' + type_of_alpha + '.csv'
    interventions_path = join(data_dir, 'us_data', name_interventions)
    name_cases = 'infections_timeseries_w_states_' + type_of_alpha + '.csv'
    cases_path = join(data_dir, 'us_data', name_cases)
    name_deaths = 'deaths_timeseries_w_states_' + type_of_alpha + '.csv'
    deaths_path = join(data_dir, 'us_data', name_deaths)

    real_cases_path = join('data', 'us_data', 'infections_timeseries_w_states.csv')
    real_deaths_path = join('data', 'us_data', 'deaths_timeseries_w_states.csv')
    real_cases_df = pd.read_csv(real_cases_path, index_col='FIPS')
    real_deaths_df = pd.read_csv(real_deaths_path, index_col='FIPS')
    
    print(generator.alphas)
    summary = {'N2':N2, 'alpha_mu':alpha_mu, 'alpha_var':alpha_var, 'alphas':generator.alphas}
    df = pd.DataFrame.from_dict(summary)
    df.to_csv(summary_path)
    
    cases_df = real_cases_df.copy()
    deaths_df = real_deaths_df.copy()

    for r in range(len(regions)):
        region = geocode[r]

        simulated_cases = all_cases[region][0:len(real_cases_df.loc[region, start_date[r]:])]
        cases_df.loc[region, start_date[r]:] = simulated_cases

        simulated_deaths = all_deaths[region][0:len(real_deaths_df.loc[region, start_date[r]:])]
        deaths_df.loc[region, start_date[r]:] = simulated_deaths

    cases_df.to_csv(cases_path)
    deaths_df.to_csv(deaths_path)
    rt_df = pd.DataFrame.from_dict(all_rt)
    rt_df.to_csv(interventions_path)
