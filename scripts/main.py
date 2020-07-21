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


def get_alpha_from_summary(df):
    """ result_df or summary from running the model. Return array of alphas

    :param df: 
    :returns: 
    :rtype: 

    """
    alpha = df.loc[[f'alpha[{i}]' for i in range(1, 9)], 'mean'].to_numpy()
    return alpha


def is_county(fips):
    """
    check if FIPS is valid
    """
    fips = str(fips).zfill(5)
    return len(fips) == 5


class MainStanModel():
    def __init__(self, args):
        self.args = args
        for k, v in args.__dict__.items():
            setattr(self, k, v)

        if isinstance(self.processing, int):
            self.processing = data_parser.Processing(self.processing)

        if self.model == 'mobility': ## to include foot traffic data
            self.use_mobility = True
        else:
            self.use_mobility = False


        self.clustering = data_parser.get_clustering(self.data_dir)
        if self.fips_list is None and self.cluster is not None:
            self.fips_list = data_parser.get_cluster(self.data_dir, self.cluster)

        stan_data, regions, start_date, geocode, weighted_fatalities = self.preprocess_data(self.M, self.mode, self.data_dir)
        if self.validation_on_county:
            train, val = self.validation_county_split(stan_data, regions, start_date, geocode, weighted_fatalities)
            stan_data, regions, start_date, geocode, weighted_fatalities = train
            print(f'Validating on: {val[1]}')

        if self.result_dir is None:
            result_df = self.run_model(stan_data, regions, start_date, geocode, weighted_fatalities)
            self.save_results(result_df, start_date, geocode)
        else:
            result_df = self.load_results(self.result_dir)

        self.make_plots()
            
        if self.validation_on_county:
            vals = self.divide_validation_counties(val)
            for val in vals:
                stan_data, regions, start_date, geocode, weighted_fatalities = val
                stan_data['alpha'] = get_alpha_from_summary(result_df)
                val_df = self.run_model(stan_data, regions, start_date, geocode, weighted_fatalities, validation=True)
                self.save_results(val_df, start_date, geocode, validation=True)
                self.make_plots(validation=True)

    def divide_validation_counties(self, val):
        """Divide the counties in 5-tuple validation arguments `val` into a list of such tuples with one county each.

        :param val: tuple containing (stan_data, regions, start_date, geocode, weighted_fatalities) as returned by self.preprocess_data
        :returns: 
        :rtype: 

        """
        stan_data, regions, start_date, geocode, weighted_fatalities = val
        vals = []
        for i, idx in enumerate(sorted(geocode.keys())):
            stan_data_ = stan_data.copy()
            stan_data_['M'] = 1
            for k in ['cases', 'deaths']:
                stan_data_[k] = stan_data[k][:, i:i + 1]
            for k in ['N', 'EpidemicStart', 'pop', 'X']:
                stan_data_[k] = stan_data[k][i:i + 1]
            
            weighted_fatalities_ = weighted_fatalities[i:i + 1]
            start_date_ = {0: start_date[idx]}
            geocode_ = {0: geocode[idx]}
            regions_ = [regions[i]]
            vals.append((stan_data_, regions_, start_date_, geocode_, weighted_fatalities_))
        return vals
        
    def get_cluster(self, region):
        """
        :param region: Either an FIPS or FIPS_CLUSTER string.
        :returns: cluster the region belongs to.
        :rtype: 

        """
        if region in self.clustering:
            return self.clustering[region]
        else:
            return int(region.split('_')[1])

    def validation_county_split(self, stan_data, regions, start_date, geocode, weighted_fatalities,
                                counties_per_cluster=1):
        """Separate into training and validation data by taking out the first county/supercounty for each cluster.

        So if cluster was specified, the validation size will be 1. If no cluster specified, there
        will be `num_clusters` counties in the validation set, one for each cluster.

        Moreover, each validation county should be it's own set, basically, to constrain the fit.

        :param stan_data: 
        :param regions: 
        :param start_date: 
        :param geocode: 
        :param weighted_fatalities: 
        :param counties_per_cluster: validation counties per cluster to withhold

        """
        # print('before split: {}, {}, {}'.format(regions[0], stan_data['pop'][0], weighted_fatalities[0, :3]))
        # print('before split: {}, {}, {}'.format(regions[-1], stan_data['pop'][-1], weighted_fatalities[-1, :3]))
        print(*[f'{k}: {v.shape if isinstance(v, np.ndarray) else v}' for k, v in stan_data.items()], sep='\n')
        print(f'regions: {regions}')
        print(f'geocode: {geocode}')
        print(f'start_date: {start_date}')
        print(f'weighted_fatalities: {weighted_fatalities}')
        counts = {}
        val_regions = []
        train_regions = []
        val_indices = []
        train_indices = []

        train_start_date = {}
        val_start_date = {}
        train_geocode = {}
        val_geocode = {}

        validx = 0
        trainidx = 0
        
        for i, region in enumerate(regions):
            c = self.get_cluster(region)
            count = counts.get(c, 0)
            if count < counties_per_cluster:
                counts[c] = count + 1
                val_regions.append(region)
                val_indices.append(i)
                val_start_date[validx] = start_date[i]
                val_geocode[validx] = geocode[i]
                validx += 1
            else:
                train_regions.append(region)
                train_indices.append(i)
                train_start_date[trainidx] = start_date[i]
                train_geocode[trainidx] = geocode[i]
                trainidx += 1
                
        print(f'val_regions: {val_regions}')
        
        train_stan_data = stan_data.copy()
        val_stan_data = stan_data.copy()
        val_stan_data['M'] = len(val_regions)
        train_stan_data['M'] = len(train_regions)

        for k in ['cases', 'deaths']:
            train_stan_data[k] = stan_data[k][:, train_indices]
            val_stan_data[k] = stan_data[k][:, val_indices]
        for k in ['N', 'EpidemicStart', 'pop', 'X']:
            train_stan_data[k] = stan_data[k][train_indices]
            val_stan_data[k] = stan_data[k][val_indices]

        train_weighted_fatalities = weighted_fatalities[train_indices]
        val_weighted_fatalities = weighted_fatalities[val_indices]

        # print('after split: {}, {}, {}'.format(
        #     train_regions[0], train_stan_data['pop'][0], train_weighted_fatalities[0, :3]))
        # print('after split: {}, {}, {}'.format(
        #     train_regions[-1], train_stan_data['pop'][-1], train_weighted_fatalities[-1, :3]))
        print(*[f'{k}: {v.shape if isinstance(v, np.ndarray) else v}' for k, v in train_stan_data.items()], sep='\n')
        print(f'train_regions: {train_regions}')
        print(f'train_geocode: {train_geocode}')
        print(f'train_start_date: {train_start_date}')
        print(f'train_weighted_fatalities: {train_weighted_fatalities}')
        
        return ((train_stan_data, train_regions, train_start_date, train_geocode, train_weighted_fatalities),
                (val_stan_data, val_regions, val_start_date, val_geocode, val_weighted_fatalities))

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
            
    def preprocess_data(self, M, mode, data_dir):
        """
        Function for preprocessing data 
        """
        if mode == 'europe':
            stan_data, regions, start_date, geocode = data_parser.get_data_europe(data_dir, show=False)
            weighted_fatalities = np.loadtxt(join(data_dir, 'europe_data', 'weighted_fatality.csv'),
                                             skiprows=1, delimiter=',', dtype=str)
            
        elif mode == 'US_county':
            stan_data, regions, start_date, geocode = data_parser.get_data(
                M, data_dir, processing=self.processing, state=False, fips_list=self.fips_list,
                validation=self.validation_withholding, supercounties=self.supercounties,
                clustering=self.clustering, mobility=self.use_mobility, load_supercounties=self.load_supercounties,
                avg_window=self.avg_window, mask_term=(self.model == 'mask'))
            weighted_fatalities = self.get_weighted_fatalities(regions)
                
        elif mode == 'US_state':
            stan_data, regions, start_date, geocode = data_parser.get_data(
                M, data_dir, processing=self.processing, state=True, fips_list=self.fips_list,
                validation=self.validation_withholding)
            wf_file = join(data_dir, 'us_data', 'state_weighted_fatality.csv')
            weighted_fatalities = np.loadtxt(wf_file, skiprows=1, delimiter=',', dtype=str)

        self.N2 = stan_data['N2']

        return stan_data, regions, start_date, geocode, weighted_fatalities

    def get_supercounties(self):
        with open(join(self.data_dir, 'us_data', 'supercounties.json'), 'r') as file:
            supercounties = json.load(file)
        return supercounties
    
    def summarize_regions(self, regions):
        supercounties = self.get_supercounties()
        count = 0
        for region in regions:
            if is_county(region):
                count += 1
            else:
                count += len(supercounties[region])
        print(f'running model on {len(regions)} regions (counties + supercounties) with {count} total counties')

    def run_model(self, stan_data, regions, start_date, geocode, weighted_fatalities, validation=False):
        """Run the model

        :param stan_data: input to the model
        :param weighted_fatalities: 
        :param regions: 
        :param start_date: 
        :param geocode: 
        :param validation: 
        :returns: 
        :rtype: 

        """

        # for k, v in stan_data.items():
        #     print(f'stan_data[{k}] = {v}')
        # print('regions:', regions)
        # print('start_date:', start_date)
        # print('geocode:', geocode)
        
        # Build a dictionary of region identifier to weighted fatality rate
        self.summarize_regions(regions)
        
        ifrs = {}
        for i in range(weighted_fatalities.shape[0]):
            ifrs[weighted_fatalities[i, 0]] = weighted_fatalities[i, -1]
        stan_data['cases'] = stan_data['cases'].astype(np.int)
        stan_data['deaths'] = stan_data['deaths'].astype(np.int)
        if self.mode[0:2] == 'US':
            if validation:
                sm = pystan.StanModel(file='stan-models/us_val.stan')
            elif self.model == 'old_alpha':
                sm = pystan.StanModel(file='stan-models/base_us.stan')
            elif self.model == 'mask':
                sm = pystan.StanModel(file='stan-models/us_mask.stan')
            elif self.model == 'pop':
                sm = pystan.StanModel(file='stan-models/us_new.stan')
            elif self.model == 'mobility':
                sm = pystan.StanModel(file='stan-models/base_us_mobility.stan')
            else:
                raise ValueError
        else:
    # Train the model and generate samples - returns a StanFit4Model
            sm = pystan.StanModel(file='stan-models/base_europe.stan')

        serial_interval = np.loadtxt(join(self.data_dir, 'us_data', 'serial_interval.csv'), skiprows=1, delimiter=',')
    # Time between primary infector showing symptoms and secondary infected showing symptoms - this is a probability distribution from 1 to 100 days
        print(serial_interval)
        print(stan_data['N2'])
        SI = serial_interval[0:stan_data['N2'],1]
        stan_data['SI'] = SI
        print(SI)
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

        fit = sm.sampling(data=stan_data, iter=self.iter, warmup=self.warmup_iter, chains=4, 
                          thin=4, control={'adapt_delta': 0.99, 'max_treedepth': self.max_treedepth})
        # fit = sm.sampling(data=stan_data, iter=2000, chains=4, warmup=10, thin=4, seed=101, control={'adapt_delta':0.9, 'max_treedepth':10})

        if validation:
            summary_dict = fit.summary(pars={'mu', 'E_deaths', 'prediction', 'Rt_adj'})
        else:
            summary_dict = fit.summary(pars={'mu', 'alpha', 'E_deaths', 'prediction', 'Rt_adj'})
            
        df = pd.DataFrame(summary_dict['summary'],
                          columns=summary_dict['summary_colnames'],
                          index=summary_dict['summary_rownames'])
        return df 

    @property
    def unique_results_path(self):
        if hasattr(self, '_unique_results_path'):
            return self._unique_results_path

        if self.result_dir is not None:
            self._unique_results_path = self.result_dir
            return self.result_dir
        
        timestamp = dt.datetime.now().strftime('%m-%d-%y_%H-%M-%S')
        unique_folder_name_list = [timestamp, str(self.mode), 'iter', str(self.iter), 'warmup',
                                   str(self.warmup_iter), 'num_counties', str(self.M), 'processing',
                                   str(data_parser.Processing(self.processing))]
        if self.validation_withholding:
            unique_folder_name_list.insert(2, 'validation_withholding')
        if self.cluster is not None:
            unique_folder_name_list.insert(3, 'cluster')
            unique_folder_name_list.insert(4, str(self.cluster))
            
        #make unique results folder
        self._unique_results_path = join(self.output_path, '_'.join(unique_folder_name_list))
        if not exists(self._unique_results_path):
            os.mkdir(self._unique_results_path)
        return self._unique_results_path

    def load_results(self, result_dir=None):
        """load the result df, and set geocodes and start date paths"""
        if result_dir is None:
            result_dir = self.unique_results_path

        self.summary_path = join(result_dir, 'summary.csv')
        self.start_dates_path = join(result_dir, 'start_dates.csv')
        self.geocode_path = join(result_dir, 'geocode.csv')
        print(f'loaded results from {self.summary_path}')
        return pd.read_csv(self.summary_path, index_col=0)
    
    def save_results(self, df, start_date, geocode, validation=False):
        """save the result dict, geocodes and start_dates into a unique folder"""
        # results example:
        #        - 05_06_2020_15_40_35_validation_iter_200_warmup_100_processing_REMOVE_NEGATIVE_VALUES
        print(f'Saving results to f{self.unique_results_path}')

        if validation:
            self.summary_path = join(self.unique_results_path, 'val_summary.csv')
            self.start_dates_path = join(self.unique_results_path, 'val_start_dates.csv')
            self.geocode_path = join(self.unique_results_path, 'val_geocode.csv')
            logfile_path = join(self.unique_results_path, 'val_logfile.txt')
        else:
            self.summary_path = join(self.unique_results_path, 'summary.csv')
            self.start_dates_path = join(self.unique_results_path, 'start_dates.csv')
            self.geocode_path = join(self.unique_results_path, 'geocode.csv')
            logfile_path = join(self.unique_results_path, 'logfile.txt')
            
        if self.validation_withholding:
            shutil.copyfile(join(self.data_dir, 'us_data', 'validation_days.csv'),
                            join(self.unique_results_path, 'validation_days.csv'))

        df.to_csv(self.summary_path, sep=',')

        if self.supercounties:
            # save a copy of the supercounties for this run
            with open(join(self.unique_results_path, 'supercounties.json'), 'w') as file:
                json.dump(self.get_supercounties(), file) 

        df_sd = pd.DataFrame(start_date, index=[0])
        df_geo = pd.DataFrame(geocode, index=[0])
        df_sd.to_csv(self.start_dates_path, sep=',')
        df_geo.to_csv(self.geocode_path, sep=',')

        with open(logfile_path, 'w') as f:
            f.write(json.dumps(self.args.__dict__))
        print('Done saving.')

    def make_plots(self, validation=False):
        """ save plots of current run"""
        print(f'Creating figures.')
        if validation and self.cluster is None:
            forecast_plots_path = join(self.unique_results_path, f'val_plots', 'forecast') 
            rt_plots_path = join(self.unique_results_path, f'val_plots', 'rt')
        elif validation and self.cluster is not None:
            forecast_plots_path = join(self.unique_results_path, f'val_plots_cluster_{self.cluster}', 'forecast') 
            rt_plots_path = join(self.unique_results_path, f'val_plots_cluster_{self.cluster}', 'rt')
        else:
            forecast_plots_path = join(self.unique_results_path, 'plots', 'forecast') 
            rt_plots_path = join(self.unique_results_path, 'plots', 'rt')

        if not exists(forecast_plots_path):
            os.makedirs(forecast_plots_path)
        if not exists(rt_plots_path):
            os.makedirs(rt_plots_path)

        # interventions_path = join(self.data_dir, 'us_data', 'interventions.csv')
        interventions_path = join(self.data_dir, 'tmp_interventions.csv')
        if self.mode == 'europe':
            forecast_plots.make_all_eu_plots(self.start_dates_path, self.geocode_path, self.summary_path,
                                             forecast_plots_path)
        elif self.mode == 'US_county':
            forecast_plots.make_all_us_county_plots(self.start_dates_path, self.geocode_path,
                                                    self.summary_path, forecast_plots_path, use_tmp=True,
                                                    avg_window=self.avg_window)
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
    parser.add_argument('--output-path', default='./results', help='directory to save the results and plots in')
    parser.add_argument('--mode', default='US_county', choices=['europe', 'US_state', 'US_county'], help='choose which data to use')
    parser.add_argument('--result-dir', default=None, help='path to result dir to load the summary.csv from instead of running the model')
    parser.add_argument('--processing', type=int, default=1, choices=[0,1,2], help=' choose the processing technique to remove negative values. \n 0 : interpolation \n 1 : replacing with 0 \n 2 : discarding regions with negative values')
    parser.add_argument('-M', default=25, type=int, help='threshold for relevant counties')
    parser.add_argument('-val-1', '--validation-withholding', action='store_true', help='whether to apply validation by withholding days')
    parser.add_argument('-val-2', '--validation-on-county', action='store_true', help='validate the model by withholding the last county (or supercounty) and learning new R_0 values on the withheld county from the alphas learned')
    parser.add_argument('--model', default='pop', choices=['old_alpha', 'new_alpha', 'pop', 'mobility', 'mask'], help='which model to use')
    parser.add_argument('--plot', action='store_true', help='add for generating plots')
    parser.add_argument('--fips-list', default=None, nargs='+', help='fips codes to run the model on')
    parser.add_argument('--cluster', default=None, type=int, help='cluster label to draw fips-list from')
    parser.add_argument('-s', '--save-tag', default='', type=str, help='tag for saving the summary, geocodes and start-dates.')
    parser.add_argument('--iter', default=1000, type=int, help='iterations for the model')
    parser.add_argument('--warmup-iter', default=500, type=int, help='warmup iterations for the model')
    parser.add_argument('--max-treedepth', default=15, type=int, help='maximum tree depth for the model')
    parser.add_argument('--supercounties', action='store_true', help='merge counties in the same state AND cluster with insufficient cases')
    parser.add_argument('--load-supercounties', action='store_true', help='load the supercounties file (don\'t overwrite it)')
    parser.add_argument('--avg-window', default=None, type=int, help='number of days to average data over for modeling')
    args = parser.parse_args()

    model = MainStanModel(args)

