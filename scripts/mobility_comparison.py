import sys
import os
import argparse 
import csv
import datetime 
import json
import pandas as pd
import numpy as np
import seaborn as sns

from os.path import join, exists
from collections import OrderedDict 
from matplotlib import pyplot as plt
from scipy import stats


# END_DATE = datetime.date(2020,5,28) old
# END_DATE = datetime.date(2020,7,7)  # new
END_DATE = datetime.date(2020,8,2)  # new

class MobilityReportParser():
    def __init__(self):
        self.report_dir = 'data/us_data/google_reports'
        self.categories, self.category_names = self.parse_google_reports(self.report_dir)
        self.start_date_ordinal = datetime.datetime.strptime(self.categories['residential'].columns[4], '%Y-%m-%d').toordinal()
        self.pop_parser = PopulationParser()

    def parse_google_reports(self, path):
        report_files = [f for f in os.scandir(path) if f.name.endswith('.csv') and len(f.name) > 25]
        categories = {}
        for idx, report in enumerate(report_files):
            df_current = pd.read_csv(report.path)
                
            # report have many NANs 
            # discard all counties that have nan values
            # df_current = df_current.dropna()
            df_current = df_current.fillna(method='backfill') # same method as IC 
            df_current = df_current.reset_index()

            categories[report_files[idx].name[:-33]] = df_current

        report_files = [f.name[:-33] for f in report_files]
        print(categories)
        return categories, report_files 

    def create_supercounties(self, supercounties):
        dict_ = {}
        counties_no_data = []
        for category_name, df in self.categories.items():
            list_of_supercounties = []
            for (supercounty_name, county_list) in supercounties.items():
                # get the population weighted average of the counties
                print(f'Creating supercounty: {supercounty_name}')
                mobility_acc = [0 for i in range(len(self.categories[self.category_names[0]].iloc[0].values.tolist()[4:]))]
                pop_acc = 0
                for county in county_list:
                    # if county == '02195':
                        # print(df[df['FIPS']==int(county)])
                    # pick the corresponding mobility timeseries
                    # handle empty series
                    try:
                        mobility_current = df[df['FIPS']==int(county)].values.tolist()[0][4:]
                    except IndexError as e:
                        print(e)
                        counties_no_data.append(county)
                        continue
                    # scale by population
                    pop_current = self.pop_parser.get_population(int(county))
                    mobility_current = [pop_current*x for x in mobility_current]
                    
                    pop_acc += pop_current
                    # this ensures that mobility_acc and mobility_current have the same length
                    mobility_acc = [mobility_current[i] + mobility_acc[i] for i in range(len(mobility_current))]
                
                # perform weigthing by dividing through the accumulated population count
                mobility_acc = [x/pop_acc for x in mobility_acc] 
                list_to_merge = [0, supercounty_name,'None','None'] + mobility_acc
                list_of_supercounties.append(list_to_merge)
                # dict_[supercounty_name] = mobility_acc
            
            print('Concatenating...')
            df_to_append = pd.DataFrame(list_of_supercounties, columns=df.columns)
            df = pd.concat([df, df_to_append])
            print(df)


class PopulationParser():
    """ Returns the POP_ESTIMATE_2018 of the census for a given fips code
    """
    def __init__(self):
        path = 'data/us_data/counties.csv' #POP_ESTIMATE_2018     
        pop = pd.read_csv(path, engine='python')
        cols_population = ['FIPS', 'POP_ESTIMATE_2018']
        self.pop = pop[cols_population]
    
    def get_population(self, fips):
        return self.pop[self.pop['FIPS']==fips].values.tolist()[0][1]


class ResultParser():
    def __init__(self, result_dir):
        assert exists(result_dir)
        self.result_dir = result_dir
        
        self.geocode, self.number_counties = self.parse_geocode(join(result_dir, 'geocode.csv'))
        self.startdates, self.startdates_dict = self.parse_start_dates(join(result_dir, 'start_dates.csv'), self.geocode) 
        self.deaths, self.infections, self.R_t = self.parse_summary(join(result_dir, 'summary.csv')) 
        self.end_date_ordinal = END_DATE.toordinal()
        self.supercounties_dict = self.parse_supercounties(join(result_dir, 'supercounties.json'))

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
        

    def parse_supercounties(self, path):
        assert exists(path)
        with open(path) as file:
            supercounties = json.load(file)
        return supercounties



class Comparison():
    """ Makes plots of the comparison between mobility reports and our expected results
    """ 

    def __init__(self, result_dir, delta, pdf):
        self.mobility_parser = MobilityReportParser()
        self.result_parser = ResultParser(result_dir)
        self.mobility_parser.create_supercounties(self.result_parser.supercounties_dict)
        self.save_to_pdf = pdf
        # print(self.mobility_parser.categories)
        self.aligned_timeseries = self.align_timeseries(self.result_parser.startdates, self.mobility_parser.categories)
        
        self.save_path = join(result_dir, 'plots', 'mobility')
        self.save_path_correlations = join(result_dir, 'plots', 'mobility_correlations')
        self.make_plots(self.aligned_timeseries, self.save_path, result_dir, self.save_path_correlations)
        self.shift_timeseries(self.save_path_correlations, self.result_parser.startdates, self.mobility_parser.categories, delta)


    def align_timeseries(self, startdates, categories, delta=0):
        """ delta is the artifical offset between the two timeseries:
            delta > 0 shift mobility into the future
            delta < 0 shift mobility into the past
        """

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
            difference = j - self.mobility_parser.start_date_ordinal - delta 
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


    def make_plots(self, timeseries_dict, save_path, result_dir, save_path_correlations):
        """ Makes Plots with the alignes mobility timeseries and the predicted deaths and infections and rt"""


        if not exists(save_path):
            os.mkdir(save_path)

        if not exists(save_path_correlations):
            os.mkdir(save_path_correlations)

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
                    current_mobility = current_mobility.values.tolist()[0][4:] # first 4 columns are descriptors
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
            fig.set_size_inches(18.5, 10.5)
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
                # some categories could be missing, fill those with NANs
                if category_name in available_categories_dict:
                    # print(f'Calculating for {category_name}')
                    # print(current_deaths)
                    # print(available_categories_dict[category_name])
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
        
        df_deaths_correlation = df_deaths_correlation.set_index('FIPS')
        df_infections_correlation = df_infections_correlation.set_index('FIPS')
        df_rt_correlation = df_rt_correlation.set_index('FIPS')

        # make plots of the pearson r values
        df_names = ['Deaths', 'Infections', 'R_t']
        csfont = {'fontsize':20}
        for k, df in enumerate([df_deaths_correlation, df_infections_correlation, df_rt_correlation]):
            print(f'Making Mobility Plot for {df_names[k]}')
            fig = plt.figure()

            # ax = df_deaths_correlation.plot.hist(bins=10)
            ax = sns.violinplot(data=df)
            ax.set_title(f'Correlation of Mobility and {df_names[k]}',fontdict=csfont)
            ax.set_ylabel('Pearson r',fontdict=csfont)
            ax.set_xlabel('Mobility Category',fontdict=csfont)

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(18)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(18)

            fig.set_size_inches(18.5, 10.5)
            if self.save_to_pdf:
                plt.savefig(join(save_path_correlations, df_names[k]+'.pdf'))
            plt.savefig(join(save_path_correlations, df_names[k]+'.png'), dpi=100)
            plt.close(fig)
        

    def shift_timeseries(self, save_path, start_dates, categories, delta=5):
        """ shift the mobility timeseries -delta/+delta around the deaths timeseries and calculate the correlation
        """
        assert (isinstance(delta, int))
        assert (delta < 20), f'delta too big'

        final_arr = {}
        for offset in range(-delta, delta+1):
            shifted_categories = self.align_timeseries(start_dates, categories, delta=offset)


            # calculate the correlation between the shifted mobility timeseries and the deaths timeseries
            for i, fips in enumerate(self.result_parser.geocode):
                print(f'Lag analysis for {fips} with offset {offset}')

                # individual_save_path = join(save_path, f'{fips}.png')

                current_start_date = self.result_parser.startdates_dict[fips]
                current_start_date_ordinal = datetime.datetime.strptime(current_start_date[0], '%m/%d/%y').toordinal()
                current_end_date = self.result_parser.end_date_ordinal
                
                timeseries_length = current_end_date - current_start_date_ordinal
                # deaths 
                current_deaths = self.result_parser.deaths
                current_deaths = current_deaths[current_deaths['FIPS'] == fips]
                current_deaths = current_deaths.values.tolist()[0][1:]
                current_deaths = current_deaths[:timeseries_length]


                # mobility
                available_categories_dict = {}
                for idx, (category_name, df_mobility) in enumerate(shifted_categories.items()):
                    # make dicts

                    # final_arr[category_name] = {}
                    if category_name not in final_arr:
                        final_arr[category_name] = {}
                    if df_mobility.FIPS.eq(fips).any():
                        current_mobility = df_mobility[df_mobility['FIPS'] == fips]
                        current_mobility = current_mobility.values.tolist()[0][4:] # first 4 columns are descriptors
                        current_mobility = current_mobility[:timeseries_length]
                        assert (len(current_deaths) == len(current_mobility)), f'Length deaths: {len(current_deaths)} || Length deaths: {len(current_deaths)}'
                        available_categories_dict[category_name] = current_mobility


                for category_name in self.mobility_parser.category_names:
                    # some categories could be missing, fill those with NANs
                    if category_name in available_categories_dict:
                        r_deaths, p_deaths = stats.pearsonr(current_deaths, available_categories_dict[category_name])
                        final_arr[category_name].setdefault(offset, []).append(r_deaths)
                    else:
                        final_arr[category_name].setdefault(offset, []).append(np.nan)

            # columns = ['FIPS'] + self.mobility_parser.category_names
            # df_deaths_correlation = pd.DataFrame(deaths_correlation_list, columns=columns)
        # prepare dataframe from dict
        df_list = []
        names_list = []
        for name, dict_ in final_arr.items():
            print(f'name: {name}: {dict_}')
            df = pd.DataFrame(dict_)
            print(df)
            df_list.append(df)
            names_list.append(name)

        # make plots
        # fig =  plt.figure()
        fig, axes = plt.subplots(3,2)
        counter = 0
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i,j].set_title(names_list[counter])
                axes[i,j].set_ylabel('Pearson R')
                axes[i,j].set_ylim([-1,1])
                sns.violinplot(data=df_list[counter], ax=axes[i,j], palette=sns.color_palette("RdBu", n_colors=(delta*2)+1))
                counter += 1
        fig.set_size_inches(18.5, 10.5)
        # plt.show()
        print(f'Saving to {join(save_path)}')
        if self.save_to_pdf:
            plt.savefig(join(save_path, 'death_lag_analysis.pdf'))
        plt.savefig(join(save_path, 'death_lag_analysis.png'),dpi=500)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir','-d', type=str, help='Directory of the result')
    parser.add_argument('--delta', type=int, default= 5, help='offset for lag analysis')
    parser.add_argument('--pdf', action='store_true', help='generate pdfs for the submission')

    args = parser.parse_args()

    comparison = Comparison(args.dir, args.delta, args.pdf)
