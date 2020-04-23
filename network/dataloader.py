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

    def __init__(self, data_dir='../data/us_data', counties='all'):
        self.data_dir = data_dir
        self.counties = counties

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

        
        self.valid_fips_list = self.get_fips_list()
        # Parsing the data
        self.min_date, self.max_date = self.available_dates()
        self.infections_arr = self.parse_infections(self.infections_path)
        self.deaths_arr = self.parse_deaths(self.deaths_path)
        
        self.google_reports_list = []
        for i, path in enumerate(self.google_traffic_paths):
            self.google_reports_list.append(self.parse_google_report(path))
            self.google_reports_arr = np.asarray(self.google_reports_list)
        

    def parse_fips_lookup(self, path):
        self.fips_to_combined_key = {}
        self.combined_key_to_fips = {}
        with open(path, 'r') as fips_file:
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
        return np.asarray(infections_list).T

    def parse_deaths(self, path):
        with open(path, 'r') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)

            available_dates = [dt.strptime(date, '%m/%d/%y').toordinal() for date in header[2:]]
            idx_min, idx_max = available_dates.index(self.min_date), available_dates.index(self.max_date)

            idx_min += 2
            idx_max += 2
            deaths_list = []
            for row in csv_reader:
                if row[0] in self.valid_fips_list:
                    deaths_list.append(row[idx_min:idx_max+1])
        return np.asarray(deaths_list).T


    def parse_google_report(self, path):
        name = path.split()[-1].split('.')[0]
        with open(path, 'r') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)

            available_dates = [dt.strptime(date, '%Y-%m-%d').toordinal() for date in header[3:]]
            idx_min, idx_max = available_dates.index(self.min_date), available_dates.index(self.max_date)

            idx_min += 3
            idx_max += 3
            google_report_list = []
            for row in csv_reader:
                if row[0] in self.valid_fips_list:
                    google_report_list.append(row[idx_min:idx_max+1])
        return np.asarray(google_report_list).T


    def get_fips_list(self):
        """ get intersection of FIPS codes"""
        df_deaths = pd.read_csv(self.deaths_path, encoding='latin1', dtype={'FIPS':str})
        df_infections = pd.read_csv(self.infections_path, dtype={'FIPS':str})
        df_google = pd.read_csv(self.google_traffic_paths[0], encoding='latin1', dtype={'FIPS':str})

        death_fips = df_deaths['FIPS'].to_list()
        infections_fips = df_infections['FIPS'].to_list()
        google_fips = df_google['FIPS'].to_list()

        valid_fips_list = list(self.fips_list)

        for fips_code in self.counties:
            MISSING_CODE_COUNTER = 0
            if fips_code not in death_fips:
                print(f'Warning: Requested FIPS code {fips_code} not in {self.deaths_path}. Skipping FIPS code')
                MISSING_CODE_COUNTER += 1

            if fips_code not in infections_fips:
                print(f'Warning: Requested FIPS code {fips_code} not in {self.infections_path}. Skipping FIPS code')
                MISSING_CODE_COUNTER += 1

            if fips_code not in google_fips:
                print(f'Warning: Requested FIPS code {fips_code} not in {self.google_traffic_paths[0]}. Skipping FIPS code')
                MISSING_CODE_COUNTER += 1

            if MISSING_CODE_COUNTER > 0:
                valid_fips_list.remove(fips_code)
        return valid_fips_list



                

    def available_dates(self):
        available_dates = {}
        with open(self.google_traffic_paths[0], 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            available_dates_google = header[3:]

            for i in range(len(available_dates_google)):
                date = available_dates_google[i]
                available_dates_google[i] = dt.strptime(date, '%Y-%m-%d').toordinal()

            available_dates['google'] = available_dates_google

        with open(self.infections_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            available_dates_infections = header[2:]

            for i  in range(len(available_dates_infections)):
                date = available_dates_infections[i]
                available_dates_infections[i] = dt.strptime(date, '%m/%d/%y').toordinal()

            available_dates['infections'] = available_dates_infections
        
        with open(self.deaths_path, 'r') as f:
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
        return min_date, max_date

        

    def __len__(self):
        return self.max_date - self.min_date + 1


    def __getitem__(self, idx):


        pass


if __name__ == '__main__':
    dataset = LSTMDataset()
    print(dataset.infections_arr[0])
    print(dataset.deaths_arr[0])
    print(dataset.google_reports_arr[0][1])
    print(len(dataset))
