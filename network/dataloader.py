import os
import csv
import numpy as np
import torch
import pandas as pd

from glob import glob
from os.path import join, exists 
from datetime import datetime as dt
from torch.utils.data import Dataset, DataLoader


class LSTMDataset(Dataset):
    """
    Dataset for LSTM. 
    Arguments:
        - data_dir : 
        - counties : either 'all' or list of FIPS codes
    
    
    Returns (through __getitem__):
        Dict with keys:
            -'infections': list of (cumulative) infections for one time point
            -'deaths': list of (cumulative) deaths for one time point
            -'interventions': list with length of counties which contains a boolean list for each 
                            county with length of the interventions, showing whether at this 
                            time point the intervention is in place (1) or not(0)
            -'densities': dictionary with keys:
                        -'density_population': list of length counties
                        -'density_housing': list of length counties
            -'mobility': list of the 6 mobility pattern that Google is tracking;
                            -list of length counties with data for pattern 1
                            :
                            -list of length counties with data for pattern 6 

    """

    def __init__(self, data_dir='../data/us_data', counties='all', split='train', retail_only=True, verbose=True):
        self.data_dir = data_dir
        self.counties = counties
        # split with factor 0.9
        assert split in ['train', 'val'], ValueError(f'Split can be train or val')
        self.split = split


        self.retail_only = retail_only
        self.verbose = verbose

        # make the fips list
        self.fips_lookup_path = join(self.data_dir, 'FIPS_lookup.csv')
        self.parse_fips_lookup(self.fips_lookup_path)
        if self.counties == 'all':
            self.fips_list = self.fips_to_combined_key.keys()
            self.counties = list(self.fips_list)


        # all the data paths
        self.interventions_path = join(self.data_dir, 'interventions.csv')
        self.infections_path = join(self.data_dir, 'infections_timeseries.csv')
        self.deaths_path = join(self.data_dir, 'deaths_timeseries.csv')
        self.counties_path = join(self.data_dir, 'counties.csv')

        self.google_traffic_paths = glob(join(self.data_dir, 'Google_traffic', '*baseline.csv'))

        # Get intersection of fips codes throughout all datasets
        self.valid_fips_list = self.get_fips_list(verbose=self.verbose)

        # Parsing the data
        self.min_date, self.max_date = self.available_dates()
        self.infections = self.parse_infections(self.infections_path)
        self.deaths = self.parse_deaths(self.deaths_path)
        self.interventions = self.parse_interventions(self.interventions_path)
        self.densities = self.parse_densities(self.counties_path)
        
        self.google_reports_list = []
        
        if retail_only:
            self.google_report_retail = self.parse_google_report(self.google_traffic_paths[3])
            self.google_report_retail = np.asarray(self.google_report_retail, dtype=np.int)
        else:
            for i, path in enumerate(self.google_traffic_paths):
                self.google_reports_list.append(self.parse_google_report(path))
        
            self.google_reports_list = list(map(list, zip(*self.google_reports_list)))
        print(f'Dataset loaded.\n {len(self)} Days of data for split: {self.split} \n {len(self.valid_fips_list)} counties selected.')
        
        
        



    def parse_fips_lookup(self, path):
        """ Creates hash tables for FIPS and the combined key: 'COUNTY_STATE' from FIPS_lookup.csv """
        self.fips_to_combined_key = {}
        self.combined_key_to_fips = {}
        with open(path, 'r', encoding='latin1') as fips_file:
            reader = csv.reader(fips_file, delimiter=',')
            next(reader)
            for row in reader:
                self.fips_to_combined_key[row[0]] = '_'.join(row[1:2])
                self.combined_key_to_fips['_'.join(row[1:])] = row[0]

    def parse_infections(self, path):
        with open(path, 'r', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)
            header = header[2:]
            available_dates = [dt.strptime(date, '%m/%d/%y').toordinal() for date in header]
            idx_min, idx_max = available_dates.index(self.min_date), available_dates.index(self.max_date)

            idx_min += 2
            idx_max += 2

            infections_list = []

            for row in csv_reader:
                if row[0] in self.valid_fips_list:
                    infections_list.append(row[idx_min:idx_max+1])
        return np.asarray(infections_list).T.astype(np.int)

    def parse_deaths(self, path):
        with open(path, 'r', encoding='latin1') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)

            available_dates = [dt.strptime(date, '%m/%d/%y').toordinal() for date in header[2:]]
            idx_min, idx_max = available_dates.index(self.min_date), available_dates.index(self.max_date)

            idx_min += 2
            idx_max += 2
            deaths_list = []

            for row in csv_reader:
                if row[0].zfill(5) in self.valid_fips_list:
                    deaths_list.append(row[idx_min:idx_max+1])
        return np.asarray(deaths_list).T.astype(np.int)


    def parse_google_report(self, path):
        with open(path, 'r', encoding='latin1') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)

            available_dates = [dt.strptime(date, '%Y-%m-%d').toordinal() for date in header[3:]]
            idx_min, idx_max = available_dates.index(self.min_date), available_dates.index(self.max_date)

            idx_min += 3
            idx_max += 3
            google_report_list = []
            length = idx_max - idx_min + 1

            # The google reports contain fips duplicates. Here only the first is considered
            valid_fips_list_copy = self.valid_fips_list .copy()
            for row in csv_reader:
                if row[0] in valid_fips_list_copy:
                    while len(row[idx_min:idx_max+1]) != length:
                        row.append('')
                    google_report_list.append(row[idx_min:idx_max+1])
                    valid_fips_list_copy.remove(row[0])
        google_report_list = list(map(list, zip(*google_report_list)))
 
        return google_report_list

    def parse_interventions(self, path):
        """ set values with NA to zero"""
        with open(path, 'r', encoding='latin1') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)

            self.date_range

            self.interventions_descriptions = header[3:]

            interventions_list = []
            # The google reports contain fips duplicates. Here only the first is considered
            valid_fips_list_copy = self.valid_fips_list .copy()
            
            for row in csv_reader:
                if row[0] in valid_fips_list_copy:
                    for i, intervention in enumerate(row):
                        if row[i] == 'NA':
                            row[i] = 0
                    interventions_list.append(row[4:])
                    valid_fips_list_copy.remove(row[0]) 
        return np.asanyarray(interventions_list).astype(np.int)

    def parse_densities(self, path):
        df = pd.read_csv(self.counties_path, encoding='latin1', dtype={'FIPS':str})
        density_pop = 'Density per square mile of land area - Population'
        density_housing = 'Density per square mile of land area - Housing units' 
        
        df = df[['FIPS',density_pop, density_housing]]
        df = df[df['FIPS'].isin(self.valid_fips_list)]

        return dict(zip(['density_population', 'density_housing'], [torch.tensor(df[density_pop].to_numpy()),
                                                     torch.tensor(df[density_housing].to_numpy())]))
    
    def get_interventions(self, idx):
        """ Returns the interventions for one time instance"""

        date = self.date_range[idx]
        return_list = []
        for row in self.interventions:
            county_interventions_bool = [1 if int(i) < date else 0 for i in row]    
            return_list.append(county_interventions_bool)
        return return_list


    def get_fips_list(self, verbose=True):
        """ Returns intersection of FIPS codes which are valid for all files"""

        df_deaths = pd.read_csv(self.deaths_path, encoding='latin1', dtype={'FIPS':str})
        df_infections = pd.read_csv(self.infections_path, encoding='latin1', dtype={'FIPS':str})
        df_google = pd.read_csv(self.google_traffic_paths[3], encoding='latin1', dtype={'FIPS':str})

        death_fips = df_deaths['FIPS'].to_list()
        infections_fips = df_infections['FIPS'].to_list()
        google_fips = df_google['FIPS'].to_list()

        death_fips = [f.zfill(5) for f in death_fips]
        infections_fips = [f.zfill(5) for f in infections_fips]
        google_fips = [f.zfill(5) for f in google_fips]
        
        # google reports contain many nans
        nan_google = df_google[df_google.isna().any(axis=1)]
        nan_fips = nan_google['FIPS'].to_list()
        nan_fips = [f.zfill(5) for f in nan_fips]
        


        # The google reports contain fips duplicates
        duplicates = []
        unique_list = []
        for i in google_fips:
            if i in unique_list:
                duplicates.append(i)
            unique_list.append(i)

        valid_fips_list = list(self.fips_list)


        for fips_code in self.counties:
            MISSING_CODE_COUNTER = 0
            if fips_code not in death_fips:
                if verbose:
                    print(f'Warning: Requested FIPS code {fips_code} not in {self.deaths_path}. Skipping FIPS code')
                MISSING_CODE_COUNTER += 1

            if fips_code not in infections_fips:
                if verbose:
                    print(f'Warning: Requested FIPS code {fips_code} not in {self.infections_path}. Skipping FIPS code')
                MISSING_CODE_COUNTER += 1

            if fips_code not in google_fips:
                if verbose:
                    print(f'Warning: Requested FIPS code {fips_code} not in {self.google_traffic_paths[3]}. Skipping FIPS code')
                MISSING_CODE_COUNTER += 1
            
            if fips_code in nan_fips:
                if verbose:
                    print(f'Warning: Requested FIPS code {fips_code} has NA values in {self.google_traffic_paths[3]}. Skipping FIPS code')
                MISSING_CODE_COUNTER += 1

            if MISSING_CODE_COUNTER > 0:
                valid_fips_list.remove(fips_code)

        print(f'From {len(self.fips_list)} requested counties {len(valid_fips_list)} are valid.')

        return valid_fips_list


    def available_dates(self):
        """ Gets the intersection of available dates across all timeseries files"""

        available_dates = {}
        with open(self.google_traffic_paths[0], 'r', encoding='latin1') as f:
            reader = csv.reader(f)
            header = next(reader)
            available_dates_google = header[3:]

            for i in range(len(available_dates_google)):
                date = available_dates_google[i]
                available_dates_google[i] = dt.strptime(date, '%Y-%m-%d').toordinal()

            available_dates['google'] = available_dates_google

        with open(self.infections_path, 'r', encoding='latin1') as f:
            reader = csv.reader(f)
            header = next(reader)
            available_dates_infections = header[2:]

            for i  in range(len(available_dates_infections)):
                date = available_dates_infections[i]
                available_dates_infections[i] = dt.strptime(date, '%m/%d/%y').toordinal()

            available_dates['infections'] = available_dates_infections
        
        with open(self.deaths_path, 'r', encoding='latin1') as f:
            reader = csv.reader(f)
            header = next(reader)
            available_dates_deaths = header[2:]
            
            for i in range(len(available_dates_deaths)):
                date = available_dates_deaths[i]
                available_dates_deaths[i] = dt.strptime(date, '%m/%d/%y').toordinal()

            available_dates['deaths'] = available_dates_deaths

        min_list = [available_dates[i][0] for i in ['google', 'infections', 'deaths']]
        max_list = [available_dates[i][-1] for i in ['google', 'infections', 'deaths']]

        max_date = min(max_list)
        min_date = max(min_list)
        self.date_range = np.arange(min_date, max_date + 1)
        # split into train and val:
        length = len(self.date_range)
        train_size = int(0.9*length)
        if self.split == 'train':
            self.date_range = self.date_range[:train_size+1]
            max_date = self.date_range[train_size]
        if self.split == 'val':
            self.date_range = self.date_range[train_size+1:]
            min_date = self.date_range[0]
        return min_date, max_date

    
    def __len__(self):
        length_total = self.max_date - self.min_date + 1
        return length_total


    def __getitem__(self, idx):
        return_dict = {}
        return_dict['infections'] = torch.tensor(self.infections[idx])
        return_dict['deaths'] = torch.tensor(self.deaths[idx])
        return_dict['interventions'] = torch.tensor(self.get_interventions(idx))
        return_dict['densities'] = self.densities

        if self.retail_only:
            return_dict['mobility'] = torch.tensor(self.google_report_retail[idx])
        # return_dict['mobility'] = torch.tensor(self.google_reports_list[idx])
        return return_dict


if __name__ == '__main__':
    dataset = LSTMDataset(data_dir='data/us_data', split='train')
    # print(dataset.infections.shape)
    # print(dataset.deaths.shape)
    # print(len(dataset.google_reports_list))
    print(type(dataset.infections[0,1]))
    print(dataset.infections[0])
    print(dataset[0])
    # print(dataset[0])
