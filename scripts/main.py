""" Run from root directory """
import os
import sys
import argparse
import json
import shutil
import pystan
import datetime as dt
import pandas as pd
import numpy as np
import data_parser
from statsmodels.distributions.empirical_distribution import ECDF
from os.path import join, exists

import forecast_plots
import plot_rt



class MainStanModel():
    def __init__(self, args):
        self.args = args
        for k, v in args.__dict__.items():
            setattr(self, k, v)

        if isinstance(self.processing, int):
            self.processing = data_parser.Processing(self.processing)

        if self.fips_list is None and self.cluster is not None:
            self.fips_list = data_parser.get_cluster(self.data_dir, self.cluster)

        stan_data, regions, start_date, geocode, weighted_fatalities = self.preprocess_data(self.M, self.mode, self.data_dir)
        result_df = self.run_model(stan_data, weighted_fatalities, regions, start_date, geocode)
        self.save_results_to_file(self.output_path, result_df, start_date, geocode)

        if self.plot:
            self.make_plots()

    def load_supercounties_fatalities(self):
        fatalities = np.loadtxt(
            join(self.data_dir, 'us_data', 'weighted_fatality_supercounties.csv'),
            skiprows=1, delimiter=',', dtype=str)
        indexing = [int(x.split('_')[1]) == self.cluster for x in fatalities[:, 0]]
        fatalities[:, 0] = [str(int(x.split('_')[0])) for x in fatalities[:, 0]]
        fatalities = fatalities[indexing]
        # fatalities = np.concatenate([fatalities[:, 0:1], np.zeros((fatalities.shape[0], 2), dtype=str), fatalities[:, 1:]], axis=1) 
        return fatalities

    def preprocess_data(self, M, mode, data_dir):
        if mode == 'europe':
            stan_data, regions, start_date, geocode = data_parser.get_data_europe(data_dir, show=False)
            weighted_fatalities = np.loadtxt(join(data_dir, 'europe_data', 'weighted_fatality.csv'), skiprows=1, delimiter=',', dtype=str)
            
        elif mode == 'US_county':
            stan_data, regions, start_date, geocode = data_parser.get_data(M, data_dir, processing=self.processing, state=False, fips_list=self.fips_list, validation=self.validation_withholding, cluster=self.cluster)
            # wf_file = join(self.data_dir, 'us_data', 'weighted_fatality.csv')
            wf_file = join(self.data_dir, 'us_data', 'weighted_fatality_new.csv')

            weighted_fatalities = np.loadtxt(wf_file, skiprows=1, delimiter=',', dtype=str)
            if self.supercounties:
                supercounty_weighted_fatalities = self.load_supercounties_fatalities()
                weighted_fatalities = np.concatenate([weighted_fatalities, supercounty_weighted_fatalities], axis=0)
                
        elif mode == 'US_state':
            stan_data, regions, start_date, geocode = data_parser.get_data(M, data_dir, processing=self.processing, state=True, fips_list=self.fips_list, validation=self.validation_withholding, supercounties=self.supercounties)
            wf_file = join(data_dir, 'us_data', 'state_weighted_fatality.csv')
            weighted_fatalities = np.loadtxt(wf_file, skiprows=1, delimiter=',', dtype=str)

        self.N2 = stan_data['N2']

        return stan_data, regions, start_date, geocode, weighted_fatalities
   

    def run_model(self, stan_data, weighted_fatalities, regions, start_date, geocode):
    # Build a dictionary of region identifier to weighted fatality rate
        ifrs = {}
        for i in range(weighted_fatalities.shape[0]):
            ifrs[weighted_fatalities[i,0]] = weighted_fatalities[i,-1]
        stan_data['cases'] = stan_data['cases'].astype(np.int)
        stan_data['deaths'] = stan_data['deaths'].astype(np.int)
        if self.mode[0:2] == 'US':
                if self.model == 'old_alpha':
                    sm = pystan.StanModel(file='stan-models/base_us.stan')
                elif self.model == 'new_alpha':
                    sm = pystan.StanModel(file='stan-models/base_us_new_alpha.stan')
                elif self.model == 'pop':
                    sm = pystan.StanModel(file='stan-models/us_new.stan')
        else:
    # Train the model and generate samples - returns a StanFit4Model
            sm = pystan.StanModel(file='stan-models/base_europe.stan')

        serial_interval = np.loadtxt(join(self.data_dir, 'serial_interval.csv'), skiprows=1, delimiter=',')
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

        all_f = np.zeros((self.N2, len(regions)))
        
        for r in range(len(regions)):
            ifr = float(ifrs[str(regions[r]).zfill(5)])

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

                all_f[:,r] = s * h

        stan_data['f'] = all_f

        fit = sm.sampling(data=stan_data, iter=self.iter, chains=4, warmup=self.warmup_iter,
                          thin=4, control={'adapt_delta': 0.9, 'max_treedepth': self.max_treedepth})
    # fit = sm.sampling(data=stan_data, iter=2000, chains=4, warmup=10, thin=4, seed=101, control={'adapt_delta':0.9, 'max_treedepth':10})

        summary_dict = fit.summary()
        df = pd.DataFrame(summary_dict['summary'],
                columns=summary_dict['summary_colnames'],
                index=summary_dict['summary_rownames'])

        return df 



    def save_results_to_file(self, results_path, df, start_date, geocode):
        """ save the result dict, geocodes and start_dates into a unique folder """
        # results example:
        #        - 05_06_2020_15_40_35_validation_iter_200_warmup_100_processing_REMOVE_NEGATIVE_VALUES 
        timestamp = dt.datetime.now().strftime('%m_%d_%y_%H_%M_%S')
        unique_folder_name_list = [timestamp, str(self.mode), 'iter', str(self.iter), 'warmup', str(self.warmup_iter),'num_counties', str(self.M), 'processing', str(data_parser.Processing(self.processing))]
        if self.validation_withholding:
            unique_folder_name_list.insert(2, 'validation_withholding')
        if self.cluster:
            unique_folder_name_list.insert(3, 'cluster')
            unique_folder_name_list.insert(4, self.cluster)


        #make unique results folder
        self.unique_results_path = join(results_path, '_'.join(unique_folder_name_list))
        os.mkdir(self.unique_results_path)
        print(f'Saving results to f{self.unique_results_path}')

        self.summary_path = join(self.unique_results_path, 'summary.csv')
        self.start_dates_path = join(self.unique_results_path, 'start_dates.csv')
        self.geocode_path = join(self.unique_results_path, 'geocode.csv')
        logfile_path = join(self.unique_results_path, 'logfile.txt')
        
        if self.validation_withholding:
            shutil.copyfile(join(self.data_dir,'us_data','validation_days.csv'), join(self.unique_results_path, 'validation_days.csv'))

        df.to_csv(self.summary_path, sep=',')

        df_sd = pd.DataFrame(start_date, index=[0])
        df_geo = pd.DataFrame(geocode, index=[0])
        df_sd.to_csv(self.start_dates_path, sep=',')
        df_geo.to_csv(self.geocode_path, sep=',')

        with open(logfile_path, 'w') as f:
            f.write(json.dumps(self.args.__dict__))
        print('Done saving.')

    

    def make_plots(self):
        """ save plots of current run"""
        print(f'Creating figures.')
        forecast_plots_path = join(self.unique_results_path, 'plots', 'forecast') 
        rt_plots_path = join(self.unique_results_path, 'plots', 'rt')
        os.makedirs(forecast_plots_path)
        os.makedirs(rt_plots_path)

        interventions_path = join(self.data_dir, 'us_data', 'interventions.csv')
        if self.mode == 'europe':
            forecast_plots.make_all_eu_plots(self.start_dates_path, self.geocode_path, self.summary_path,
                                             forecast_plots_path)
        elif self.mode == 'US_county':
            forecast_plots.make_all_us_county_plots(self.start_dates_path, self.geocode_path,
                                                    self.summary_path, forecast_plots_path, use_tmp=True)
            plot_rt.make_all_us_plots(self.summary_path, self.geocode_path, self.start_dates_path,
                                      interventions_path, rt_plots_path, state_level=False)
        elif self.mode == 'US_state':
            forecast_plots.make_all_us_states_plots(self.start_dates_path, self.geocode_path,
                                                    self.summary_path, forecast_plots_path)
            plot_rt.make_all_us_plots(self.summary_path, self.geocode_path, self.start_dates_path,
                                      interventions_path, rt_plots_path, state_level=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='./data/', help='directory for the data')
    parser.add_argument('--output_path', default='./results', help='directory to save the results and plots in')
    parser.add_argument('--mode', default='US_county', choices=['europe', 'US_state', 'US_county'], help='choose which data to use')
    parser.add_argument('--processing', type=int, default=1, choices=[0,1,2], help=' choose the processing technique to remove negative values. \n 0 : interpolation \n 1 : replacing with 0 \n 2 : discarding regions with negative values')
    parser.add_argument('-M', default=25, type=int, help='threshold for relevant counties')
    parser.add_argument('-val-1','--validation_withholding', action='store_true', help='whether to apply validation by withholding days')
    parser.add_argument('--model', default='pop', choices=['old_alpha', 'new_alpha', 'pop'], help='which model to use')
    parser.add_argument('--plot', action='store_true', help='add for generating plots')
    parser.add_argument('--fips-list', default=None, nargs='+', help='fips codes to run the model on')
    parser.add_argument('--cluster', default=None, type=int, help='cluster label to draw fips-list from')
    parser.add_argument('-s', '--save-tag', default='', type=str, help='tag for saving the summary, geocodes and start-dates.')
    parser.add_argument('--iter', default=200, type=int, help='iterations for the model')
    parser.add_argument('--warmup-iter', default=100, type=int, help='warmup iterations for the model')
    parser.add_argument('--max-treedepth', default=10, type=int, help='maximum tree depth for the model')
    parser.add_argument('--supercounties', action='store_true', help='merge counties in the same state AND cluster with insufficient cases')
    args = parser.parse_args()

    model = MainStanModel(args)

