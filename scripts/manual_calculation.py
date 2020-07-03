import sys
import os 
import datetime
import argparse
import pandas as pd
import numpy as np

from os.path import join, exists
from statsmodels.distributions.empirical_distribution import ECDF

import data_parser

class ManualCalculator():
    """
    Maricopa County, AZ FIPS: 04013
    """
    def __init__(self, data_dir, results_path):
        assert(exists(data_dir))
        self.data_dir = data_dir
        self.first_date_ordinal = datetime.date(2020,1,22).toordinal()

        self.cases, self.deaths = self.parse_jhu_data(self.data_dir)
        self.alpha_list, self.mu = self.parse_parameters(results_path)
        self.geocode_list, self.number_counties = self.parse_geocode(results_path)
        self.start_dates_list, self.start_dates_dict = self.parse_start_dates(results_path, self.geocode_list)

        self.cases, self.deaths = self.align_timeseries(self.start_dates_list, self.cases, self.deaths, self.geocode_list, self.first_date_ordinal)

        stan_data, regions, weighted_fatalities = self.get_data(self.number_counties, self.data_dir, ['04013'])
        stan_data['X'] = self.repeal_lockdown_date(stan_data['X'])

        self.calculate_results(stan_data, weighted_fatalities, regions, self.alpha_list, self.mu[self.geocode_list.index(4013)])

    def parse_jhu_data(self, path, fips_list=['04013']):
        deaths_path = join(path, 'us_data', 'deaths_timeseries_updated.csv')
        cases_path = join(path, 'us_data', 'infections_timeseries_updated.csv')

        deaths_df = pd.read_csv(deaths_path)
        cases_df = pd.read_csv(cases_path)

        # pick Maricopa
        cases =  cases_df.loc[cases_df['FIPS'].isin(fips_list)] 
        deaths =  deaths_df.loc[deaths_df['FIPS'].isin(fips_list)] 

        return cases, deaths


    def parse_parameters(self, path):
        summary = pd.read_csv(join(path, 'summary.csv'), engine='python')
        parameters = summary[summary['Unnamed: 0'].str.contains('alpha')].values.tolist()
        parameters = [i[1] for i in parameters]
        
        mu = summary[summary['Unnamed: 0'].str.contains('mu')].values.tolist()
        mu = [i[1] for i in mu]

        return parameters, mu

    def parse_start_dates(self, path, geocode_list):
        path = join(path, 'start_dates.csv')
        assert exists(path)
        df = pd.read_csv(path, engine='python')
        start_dates_list = df.values.tolist()[0][1:]
        start_dates_dict = df.to_dict('list')

        del start_dates_dict['Unnamed: 0']
        assert (len(geocode_list) == len(start_dates_dict)), f'Length geocode: {len(geocode_list)} || Length start_dates_dict: {len(start_dates_dict)}'
        for idx in range(len(start_dates_dict)):
            start_dates_dict[geocode_list[idx]] = start_dates_dict.pop(str(idx))
        return start_dates_list,start_dates_dict


    def align_timeseries(self, startdates, cases, deaths, geocode, first_date, delta=0):
        """ delta is the artifical offset between the two timeseries:
            delta > 0 shift mobility into the future
            delta < 0 shift mobility into the past
        """

        #grab all available fips time series
        parsed_start_dates = []
        for i in range(self.number_counties):
            current_start_date = startdates[i]
            current_start_date_ordinal = datetime.datetime.strptime(current_start_date, '%m/%d/%y').toordinal()
            parsed_start_dates.append(current_start_date_ordinal)
        
        # find the difference between mobility start date and result start date
        differences = []
        for j in parsed_start_dates:
            difference = j - first_date - delta 
            differences.append(difference)

        # make dict fips to difference
        self.fips_to_difference_dict = dict(zip(geocode, differences))
        # print(fips_to_difference_dict)
        df_list =[cases, deaths]

        for i in range(len(df_list)):
            df_list[i] = df_list[i][df_list[i]['FIPS'].isin(geocode)]
            # shift each row accordingly
            fips_values_list = df_list[i]['FIPS'].values.tolist()
            # print(df_list[i])
            for idx, fips in enumerate(fips_values_list):
                print(df_list[i].iloc[idx,2:].values)
                if self.fips_to_difference_dict[fips] > 0:
                    print(f'Shifting fips: {fips} by {self.fips_to_difference_dict[fips]}')
                    df_list[i].iloc[idx, 2:] = df_list[i].iloc[idx, 2:].shift(-self.fips_to_difference_dict[fips])
            print(f'length of FIPS list: {len(fips_values_list)}')

        cases = np.array(df_list[0].values.tolist()[0][2:])
        deaths = np.array(df_list[1].values.tolist()[0][2:])
        return cases, deaths 

    
    def parse_geocode(self, path):
        path = join(path, 'geocode.csv')
        assert exists(path)
        df = pd.read_csv(path, engine='python')
        # First entry is 0
        df = df.values.tolist()[0][1:]
        return df, len(df)

    def get_data(self, M, data_dir, fips_list, validation=False, supercounties=None, clustering=None, mobility=False):
        stan_data, regions, start_date, geocode = data_parser.get_data(
            M, data_dir, processing=1, state=False, fips_list=fips_list,
            validation=validation, supercounties=supercounties, clustering=clustering, mobility=mobility)
        weighted_fatalities = self.get_weighted_fatalities(regions)

        return stan_data, regions, weighted_fatalities

    def get_weighted_fatalities(self, regions):
        county_weighted_fatalities = np.loadtxt(
            join(self.data_dir, 'us_data', 'weighted_fatality_new.csv'),
            skiprows=1, delimiter=',', dtype=str)
        supercounty_weighted_fatalities = np.loadtxt(
            join(self.data_dir, 'us_data', 'weighted_fatality_supercounties.csv'),
            skiprows=1, delimiter=',', dtype=str)
        region_to_weights = dict(zip(county_weighted_fatalities[:, 0], county_weighted_fatalities))
        region_to_weights.update(dict(zip(supercounty_weighted_fatalities[:, 0], supercounty_weighted_fatalities)))
                
        weighted_fatalities = []
        for region in regions:
            try:
                weighted_fatalities.append(region_to_weights[region])
            except KeyError:
                raise RuntimeError(f'No weighted fatality for {region}. Maybe update weighted fatalities?')
                
        return np.stack(weighted_fatalities)
    
            
    def repeal_lockdown_date(self, covariates):
        # for 04013: https://azgovernor.gov/file/34899/download?token=3XTbfXMX 
        # May 16th 
        repeal_date_ordinal = datetime.date(2020,5,16).toordinal()
        
        position = repeal_date_ordinal - self.first_date_ordinal - self.fips_to_difference_dict[4013]
        
        # pick the stay at home order covariate
        stay_at_home_order = covariates[0].T[0]
        stay_at_home_order[position:] = [0] * (120-position)

        covariates[0].T[0] = stay_at_home_order

        return covariates   


        
    def calculate_results(self, stan_data, weighted_fatalities, regions, alpha_list, mu_list):
        ifrs = {}
        for i in range(weighted_fatalities.shape[0]):
            ifrs[weighted_fatalities[i, 0]] = weighted_fatalities[i, -1]
        stan_data['cases'] = stan_data['cases'].astype(np.int)
        stan_data['deaths'] = stan_data['deaths'].astype(np.int)

        serial_interval = np.loadtxt(join(self.data_dir, 'us_data', 'serial_interval.csv'), skiprows=1, delimiter=',')
    # Time between primary infector showing symptoms and secondary infected showing symptoms - this is a probability distribution from 1 to N2 days
        N2 = stan_data['N2']
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
            ifr = float(ifrs[str(regions[r]).zfill(5)])
            print(ifr)

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
        print(f'SI: {SI} || shape {SI.shape}') # gamma
        print(f'F: {all_f} || shape {all_f.shape}') # pi
        mu = np.array(mu_list, dtype=np.float)
        cases = stan_data['cases']
        print(cases)
        population = 4800000

        predicted_cases = []
        predicted_deaths = [] 
        R_t = []
        new_cases = np.empty_like(cases)
        interventions = np.transpose(stan_data['X'], (0,2,1))
        alpha = np.array(alpha_list, dtype=np.float)
        R_t = mu * np.exp(- np.dot(alpha, interventions)) 
        # cases = (1- np.cumsum(cases)/population) * R_t * np.cumsum(np.dot(cases.T, SI))
        print(f'R_t shape: {R_t.shape}')

        for i in range(len(cases)):
            print(f' Adjustement factor: {1 - np.sum(cases[:i+1]/ population)}')
            print(f'R_t: {R_t.T[i]}')
            print(f'{np.dot(cases.T, SI)}')

            new_cases[i] = (1-(np.sum(cases[:i]/ population)))* R_t.T[i] * np.sum(np.dot(cases.T, SI)[:i])

            print(new_cases[i]) 

        print(new_cases)


        






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/')
    parser.add_argument('--r-path', default='results/national_no_supercounty_no_validation')

    args = parser.parse_args()

    calc = ManualCalculator(args.data_dir, args.r_path)
