import sys
import os
import argparse 
import csv
import datetime 
import pandas as pd
import numpy as np

from os.path import join, exists
from collections import OrderedDict 
from matplotlib import pyplot as plt
from scipy import stats




class MobilityReportParser():
    def __init__(self):
        self.report_dir = 'data/us_data/google_reports'
        self.categories, self.category_names = self.parse_google_reports(self.report_dir)
        self.start_date_ordinal = datetime.datetime.strptime(self.categories['residential'].columns[4], '%Y-%m-%d').toordinal()

    def parse_google_reports(self, path):
        report_files = [f for f in os.scandir(path) if f.name.endswith('.csv') and len(f.name) > 25]
        categories = {}
        for idx, report in enumerate(report_files):
            df_current = pd.read_csv(report.path)
                
            # report have many NANs 
            # discard all counties that have nan values
            df_current = df_current.dropna()
            df_current = df_current.reset_index()

            categories[report_files[idx].name[:-33]] = df_current

        report_files = [f.name[:-33] for f in report_files]
        return categories, report_files 


class ResultParser():
    def __init__(self, result_dir):
        assert exists(result_dir)
        self.result_dir = result_dir
        
        self.geocode, self.number_counties = self.parse_geocode(join(result_dir, 'geocode.csv'))
        self.startdates, self.startdates_dict = self.parse_start_dates(join(result_dir, 'start_dates.csv'), self.geocode) 
        self.deaths, self.infections, self.R_t = self.parse_summary(join(result_dir, 'summary.csv')) 
        self.end_date_ordinal = datetime.date(2020,5,28).toordinal()

    def parse_summary(self, path):
        assert exists(path)
        df = pd.read_csv(path)

        # R_t dataframe
        df_r = df[df['Unnamed: 0'].str.match('Rt_adj')]
        df_r = df_r.reset_index(drop=True)
        
        # only keep the mean column
        r_columns = df_r.columns.values
        df_r = df_r.drop(r_columns[2:], axis=1)
        list_of_r_series = []
        for i in range(1, self.number_counties+1):
            current_series = df_r[df_r['Unnamed: 0'].str.endswith(','+str(i) +']')]
            current_series.reset_index(drop=True, inplace=True)
            list_of_r_series.append(current_series.T[1:])
        
        # make df from list
        df_r = pd.concat(list_of_r_series, axis=0, ignore_index=True)
        df_geocode = pd.DataFrame(self.geocode, columns=['FIPS'])
        df_r = pd.concat([df_geocode, df_r], axis=1)
        # print(list_of_r_series[0][0])

        # deaths dataframe
        df_d = df[df['Unnamed: 0'].str.match('E_deaths')]
        df_d = df_d.reset_index(drop=True)
        
        # only keep the mean column
        d_columns = df_d.columns.values
        df_d = df_d.drop(d_columns[2:], axis=1)
        list_of_d_series = []
        for i in range(1, self.number_counties+1):
            current_series = df_d[df_d['Unnamed: 0'].str.endswith(','+str(i) +']')]
            current_series.reset_index(drop=True, inplace=True)
            list_of_d_series.append(current_series.T[1:])
        
        # make df from list
        df_d = pd.concat(list_of_d_series, axis=0, ignore_index=True)
        df_geocode = pd.DataFrame(self.geocode, columns=['FIPS'])
        df_d = pd.concat([df_geocode, df_d], axis=1)
        # print(list_of_r_series[0][0])
        print(df_d)
        # infections dataframe
        df_i = df[df['Unnamed: 0'].str.match('prediction')]
        df_i = df_i.reset_index(drop=True)
        
        # only keep the mean column
        i_columns = df_i.columns.values
        df_i = df_i.drop(i_columns[2:], axis=1)
        list_of_i_series = []
        for i in range(1, self.number_counties+1):
            current_series = df_i[df_i['Unnamed: 0'].str.endswith(','+str(i) +']')]
            current_series.reset_index(drop=True, inplace=True)
            list_of_i_series.append(current_series.T[1:])
        
        # make df from list
        df_i = pd.concat(list_of_i_series, axis=0, ignore_index=True)
        df_geocode = pd.DataFrame(self.geocode, columns=['FIPS'])
        df_i = pd.concat([df_geocode, df_i], axis=1)
        # print(list_of_r_series[0][0])
        return df_d, df_i, df_r 


    def parse_start_dates(self, path, geocode_list):
        assert exists(path)
        df = pd.read_csv(path)
        start_dates_list = df.values.tolist()[0][1:]
        start_dates_dict = df.to_dict('list')

        del start_dates_dict['Unnamed: 0']
        assert (len(geocode_list) == len(start_dates_dict)), f'Length geocode: {len(geocode_list)} || Length start_dates_dict: {len(start_dates_dict)}'
        for idx in range(len(start_dates_dict)):
            start_dates_dict[geocode_list[idx]] = start_dates_dict.pop(str(idx))
        return start_dates_list,start_dates_dict


    
    def parse_geocode(self, path):
        assert exists(path)
        df = pd.read_csv(path)
        # First entry is 0
        df = df.values.tolist()[0][1:]
        return df, len(df)



class Comparison():
    def __init__(self, result_dir):
        self.mobility_parser = MobilityReportParser()
        self.result_parser = ResultParser(result_dir)
        print(self.mobility_parser.categories)
        self.aligned_timeseries = self.align_timeseries(self.result_parser.startdates, self.mobility_parser.categories)
        
        self.save_path = join(result_dir, 'plots', 'mobility')
        self.make_plots(self.aligned_timeseries, self.save_path, result_dir)


    def align_timeseries(self, startdates, categories):


        aligned_categories = categories.copy()

        #grab all available fips time series
        parsed_start_dates = []
        for i in range(self.result_parser.number_counties):
            current_start_date = self.result_parser.startdates[i]
            current_start_date_ordinal = datetime.datetime.strptime(current_start_date, '%m/%d/%y').toordinal()
            parsed_start_dates.append(current_start_date_ordinal)
        
        # find the difference between mobility start date and result start date
        differences = []
        for j in parsed_start_dates:
            difference = j - self.mobility_parser.start_date_ordinal
            differences.append(difference)

        # make dict fips to difference
        fips_to_difference_dict = dict(zip(self.result_parser.geocode, differences))
        # print(fips_to_difference_dict)

        for idx, (mobility_category_name, df_mobility) in enumerate(aligned_categories.items()):
            df_mobility = df_mobility[df_mobility['FIPS'].isin(self.result_parser.geocode)]
            # shift each row accordingly
            fips_values_list = df_mobility['FIPS'].values.tolist()
            # print(df_mobility)
            for idx, fips in enumerate(fips_values_list):
                # print(df_mobility.iloc[idx+1,4:])
                if fips_to_difference_dict[fips] > 0:
                    df_mobility.iloc[idx, 4:] = df_mobility.iloc[idx, 4:].shift(-fips_to_difference_dict[fips])
                # print(df_mobility.iloc[idx+1, 4:])
            aligned_categories[mobility_category_name] = df_mobility
            print(f'length of FIPS list: {len(fips_values_list)}')
            # print(df_mobility)
        

        return aligned_categories   


    def make_plots(self, timeseries_dict, save_path, result_dir):
        """ Makes Plots with the alignes mobility timeseries and the predicted deaths and infections and rt"""


        if not exists(save_path):
            os.mkdir(save_path)

        deaths_correlation_list = []
        infections_correlation_list = []
        rt_correlation_list = []

        for i, fips in enumerate(self.result_parser.geocode):
            print(f'Making Plots for {fips}')
            individual_save_path = join(save_path, f'{fips}.png')

            current_start_date = self.result_parser.startdates_dict[fips]
            current_start_date_ordinal = datetime.datetime.strptime(current_start_date[0], '%m/%d/%y').toordinal()
            current_end_date = self.result_parser.end_date_ordinal
            
            timeseries_length = current_end_date - current_start_date_ordinal

            # data
            # prediction 
            current_infections = self.result_parser.infections
            current_infections = current_infections[current_infections['FIPS'] == fips].values.tolist()[0][1:]
            current_infections = current_infections[:timeseries_length] # until last date 

            # deaths 
            current_deaths = self.result_parser.deaths
            current_deaths = current_deaths[current_deaths['FIPS'] == fips]
            current_deaths = current_deaths.values.tolist()[0][1:]
            current_deaths = current_deaths[:timeseries_length]

            # Rt 
            current_rt = self.result_parser.R_t
            current_rt = current_rt[current_rt['FIPS']== fips].values.tolist()[0][1:]
            current_rt = current_rt[:timeseries_length]

            # Handle the figure
            fig = plt.figure()
            ax = fig.add_subplot(111)
            

            # Plots
            # timeseries deaths, infections, R_t
            x = np.linspace(0, timeseries_length, timeseries_length)
            assert (len(x) == len(current_infections[:timeseries_length])), f'Length x: {len(x)} || Length deaths: {len(current_deaths)}'
            assert (len(x) == len(current_deaths[:timeseries_length])), f'Length x: {len(x)} || Length deaths: {len(current_deaths)}'
            assert (len(x) == len(current_rt[:timeseries_length])), f'Length x: {len(x)} || Length deaths: {len(current_deaths)}'
            # ax.scatter(x, current_infections[:timeseries_length], label='Infections' )
            ax.plot(x, current_deaths[:timeseries_length], label='Deaths', linewidth=3)
            ax.plot(x, current_rt[:timeseries_length], label='R_t', linewidth=3)


            # 2nd y axis
            ax2 = ax.twinx() # share x axis
            ax2.set_ylabel('Mobilities')
            




            # Mobilities 
            available_categories_dict = {}
            for idx, (category_name, df_mobility) in enumerate(timeseries_dict.items()):
                if df_mobility.FIPS.eq(fips).any():
                    current_mobility = df_mobility[df_mobility['FIPS'] == fips]
                    current_mobility = current_mobility.values.tolist()[0][4:]
                    current_mobility = current_mobility[:timeseries_length]
                    assert (len(x) == len(current_mobility)), f'Length x: {len(x)} || Length deaths: {len(current_deaths)}'
                    available_categories_dict[category_name] = current_mobility
                    ax2.plot(x, current_mobility, '--', label=category_name)

            


            title = f'Mobility Comparison for FIPS: {fips}'
            ax.set_title(title)
            ax.set_ylabel('Model Results')
            ax.legend(loc='best')
            ax2.legend(loc='best')
            fig.tight_layout()
            plt.savefig(individual_save_path, dpi=100)
            plt.close(fig)

        
            # calculate correlation
            deaths_correlation_list.append([])
            infections_correlation_list.append([])
            rt_correlation_list.append([])
            deaths_correlation_list[i].append(fips)
            infections_correlation_list[i].append(fips)
            rt_correlation_list[i].append(fips)
            for category_name in self.mobility_parser.category_names:
                if category_name in available_categories_dict:
                    r_deaths, p_deaths = stats.pearsonr(current_deaths, available_categories_dict[category_name])
                    r_infections, p_infections = stats.pearsonr(current_infections, available_categories_dict[category_name])
                    r_rt, p_rt = stats.pearsonr(current_rt, available_categories_dict[category_name])
                    deaths_correlation_list[i].append(r_deaths)
                    infections_correlation_list[i].append(r_infections)
                    rt_correlation_list[i].append(r_rt)
                else:
                    deaths_correlation_list[i].append(np.nan)
                    infections_correlation_list[i].append(np.nan)
                    rt_correlation_list[i].append(np.nan)

        # save the correlation results
        columns = ['FIPS'] + self.mobility_parser.category_names
        df_deaths_correlation = pd.DataFrame(deaths_correlation_list, columns=columns)
        df_infections_correlation = pd.DataFrame(infections_correlation_list, columns=columns)
        df_rt_correlation = pd.DataFrame(rt_correlation_list, columns=columns)

        df_deaths_correlation.to_csv(join(result_dir,'deaths_correlation.csv'), index=False)
        df_infections_correlation.to_csv(join(result_dir,'infections_correlation.csv'), index=False)
        df_rt_correlation.to_csv(join(result_dir,'rt_correlation.csv'), index=False)
        print(df_deaths_correlation)
        print(df_infections_correlation)
        print(df_rt_correlation)

            
        



        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir','-d', type=str, help='Directory of the result')

    args = parser.parse_args()

    comparison = Comparison(args.dir)
