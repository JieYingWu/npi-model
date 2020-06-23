import os 
import os 
import csv
import json
import argparse
import datetime as dt 
import pandas as pd 
import numpy as np
import scipy


from os.path import join, exists
from statsmodels.stats import weightstats as stests
from scipy import stats
from matplotlib import pyplot as plt

# plt.style.use('seaborn-darkgrid')
# plt.rcParams["font.family"] = "Helvetica"


class ValidationResult():
    """
        Compares two distributions via the Pearson Correlation Coefficient and saves the resulting score and p value in a csv file

        Usage:
            - specify 2 result directories to compare after the ---results_path argument
            -> the resulting 'comparison.csv' and 'final_comparison.csv' is saved in the first directory

        Example:
        python scripts/ValidationResult.py --results-path results/05_13_20_23_54_17_US_county_iter_200_warmup_100_num_counties_100_processing_Processing.REMOVE_NEGATIVE_VALUES/ results/05_14_20_01_27_52_US_county_validation_withholding_iter_200_warmup_100_num_counties_100_processing_Processing.REMOVE_NEGATIVE_VALUES/
    
        
    """
    def __init__(self, args):
        self.args = args
        for k, v in args.__dict__.items():
            setattr(self, k, v)

        self.summary_list = []
        for path in self.results_path:
            self.summary_list.append(self.parse_summary(path))
        

        # always save results into the first directory
        self.save_path = self.results_path[0]
        geocode_1 = self.get_geocode(self.results_path[0])
        geocode_2 = self.get_geocode(self.results_path[1])

        assert geocode_1 == geocode_2, ValueError('geocode.csv for the two paths not similar')

        self.geocode = geocode_1
        print(f'Length of geocode: {len(self.geocode)}')
        self.num_counties = self.get_num_counties(self.results_path[0])
        # print(f'Number of counties: {self.num_counties}')
        self.start_dates = self.get_start_dates(self.results_path[0])
        self.start_date, self.end_date, self.validation_days_list, self.validation_days_dict = self.get_validation_days(self.results_path[1])
        

        self.make_validation_plot(self.save_path, self.summary_list[0], self.summary_list[1], self.validation_days_list)

    def parse_summary(self, path):
        with open(join(path, 'summary.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)

            # read parameters
            # They are order in the summary file in the same way, i.e. from 1 to M
            mu = [] # length M
            kappa = [] # length 1 
            alpha = [] # length M 
            predicted_deaths = {} # M key-val pairs with length N
            rt_adj = {}  # M key-val pairs with length N
            
            self.parameter_name_list = ['mu', 'kappa', 'alpha', 'predicted_deaths', 'rt_adj']
            for row in reader:
                if row[0][:2] == 'mu':
                    mu.append(row[1:])
                
                if row[0][:5] == 'kappa':
                    kappa.append(row[1:])

                if row[0][:6] == 'alpha[':
                    alpha.append(row[1:])

                if row[0][:9] == 'E_deaths[':
                    pos = row[0][row[0].find('[')+1:row[0].find(']')]
                    t, m = pos.split(',')
                    predicted_deaths.setdefault(int(m), []).append(row[1:])
                
                if row[0][:6] == 'Rt_adj':
                    pos = row[0][row[0].find('[')+1:row[0].find(']')]
                    t, m = pos.split(',')
                    rt_adj.setdefault(int(m), []).append(row[1:])
                        

        return mu, kappa, alpha, predicted_deaths, rt_adj 



    def get_num_counties(self, path):
        with open(join(path, 'logfile.txt'),'r') as f:
            args = json.load(f)
        return args['M']   


    def get_start_dates(self, path):
        with open(join(path,'start_dates.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            start_dates = next(reader)[1:]
        return start_dates

    def get_geocode(self, path):
        with open(join(path,'geocode.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            geocode = next(reader)[1:]
        geocode = [i.zfill(5) for i in geocode]
        return geocode


    def get_validation_days(self, path):
        validation_days_dict = {}
        validation_days_list = []
        with open(join(path, 'validation_days.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            start_date, end_date = next(reader)
            next(reader)
            for row in reader:
                validation_days_dict[row[0]] = row[1:]
                validation_days_list.append([int(i) for i in row[1:]])


        return start_date, end_date, validation_days_list, validation_days_dict

                    


    def recursive_len(self, item):
        if type(item) == list:
            return sum(self.recursive_len(subitem) for subitem in item)
        else:
            return 1
    
    def make_validation_plot(self, path, summary1, summary2, validation_days_list):
        validation_save_path = join(path, 'plots', 'validation')
        
        if not exists(validation_save_path):
            os.mkdir(validation_save_path)

        validation_days_list_length  = self.recursive_len(validation_days_list) #sum([len(i) for i in validation_days_list])
        
        regular_list = []
        validation_list = []
        regular_mean_list = []
        validation_mean_list = []
        for i in range(len(summary1)-1):
            if  isinstance(summary1[i], dict) and 'death' in self.parameter_name_list[i]:
                assert (len(validation_days_list) == len(summary1[i])), f'Validation days have length {len(validation_days_list)} and summary: {len(summary1[i])}'
                for idx, ((key1, val1), (key2, val2)) in enumerate(zip(summary1[i].items(), summary2[i].items())): # length number counties
                    assert (len(val1) == len(val2)), f'summaries have different lengths'
                    validation_days_current = validation_days_list[idx]
                    regular_mean_list.append([])
                    validation_mean_list.append([])
                    for value_list in [val1, val2]:
                        for j in range(len(value_list)):
                            adjusted_date = dt.datetime.strptime(self.start_dates[idx],'%m/%d/%y').toordinal() - dt.datetime.strptime(self.start_date, '%m/%d/%y').toordinal() + j 
                            # print(f'county number: {idx} || date {adjusted_date}')
                            if adjusted_date in validation_days_current:
                                #print(idx,j,adjusted_date)
                                if value_list == val1:
                                    regular_list.append(value_list[j])
                                    regular_mean_list[idx].append(value_list[j][0]) # only take the mean (first value)
                                else:
                                    validation_list.append(value_list[j])
                                    validation_mean_list[idx].append(value_list[j][0]) # only take the mean (first value)
                    
                    regular_mean_list[idx] = np.mean(np.array(regular_mean_list[idx]).astype(np.float))
                    validation_mean_list[idx] = np.mean(np.array(validation_mean_list[idx]).astype(np.float))

        print(f'Expected length of validation_days_list: {validation_days_list_length} \nActual length of regular list: {len(regular_list)}') 
        print(f'Expected length of validation_days_list: {validation_days_list_length} \nActual length of validation list: {len(validation_list)}') 
        

        # Prepare arrays 
        regular_arr = np.array(regular_list).astype(np.float)[:,0]
        validation_arr = np.array(validation_list).astype(np.float)[:,0]
        regular_mean_arr = np.array(regular_mean_list)
        validation_mean_arr = np.array(validation_mean_list)
        
        idx_reg = np.argwhere(np.isnan(regular_mean_arr))
        idx_val = np.argwhere(np.isnan(validation_mean_arr))
        
        assert idx_reg.all() == idx_val.all()   
        
        regular_mean_arr = regular_mean_arr[~np.isnan(regular_mean_arr)]
        validation_mean_arr = validation_mean_arr[~np.isnan(validation_mean_arr)]
        # similarity measures
        pearson, pval = stats.pearsonr(regular_arr, validation_arr)
        pearson_aggregated, pval_aggregated = stats.pearsonr(regular_mean_arr, validation_mean_arr)
        
        l1 = np.linalg.norm((regular_arr - validation_arr), ord=1)
        l1_aggregated = np.linalg.norm((regular_mean_arr - validation_mean_arr), ord=1)
        print(f'L1: {l1}|| L1_agg: {l1_aggregated}')
        print(f'R: {pearson}|| R_agg: {pearson_aggregated}')
        

        with open(join(self.save_path,'pearson.csv'), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([f'Length of non aggregated array {len(regular_arr)}']) 
            writer.writerow([f'Length of  aggregated array {len(regular_mean_arr)}']) 
            writer.writerow(['Pearson R value regular', 'Pearson R value aggregated'])
            writer.writerow([pearson, pearson_aggregated])
            writer.writerow(['pvalues'])
            writer.writerow([pval, pval_aggregated])
            writer.writerow(['regular', 'validation', 'regular_aggregated', 'validation aggregated'])
            
            for i in range(len(regular_arr)):
                if i < len(validation_mean_arr):
                    arr = [regular_arr[i], validation_arr[i], regular_mean_arr[i], validation_mean_arr[i]]
                else: 
                    arr = [regular_arr[i], validation_arr[i]]
                writer.writerow(arr)



        # Plotting of 3 day forecast
        individual_save_path = join(validation_save_path, f'ValidationPlot.png')
        

        fig = plt.figure()
        csfont = {'fontsize':20}
        #fig.add_axes()
        ax = fig.add_subplot(111)
        #ax2 = fig.add_subplot(122)
        rasterized = False
        ax.set(xlim=[0.5,500],ylim=[0.5,500])

        max_value = int(max(max(regular_arr), max(validation_arr))) + 1000
        min_value = int(min(min(regular_arr), min(validation_arr)))
        t = np.linspace(min_value, max_value, max_value)
        ax.plot(t, t, rasterized=rasterized, label='Optimal Fit',linewidth=2, color='red')
        ax.plot(regular_arr, validation_arr, '.', rasterized=rasterized, label='Withheld Days', color='black')

        title = 'Validation Plot'
        ax.set_title(title, fontdict = csfont)
        ax.set_ylabel('Validation Sample', fontdict = csfont)
        ax.set_xlabel('Regular Sample', fontdict=csfont)
        fig.text(.5, .05, '(a)', ha='center', fontdict=csfont)
        ax.set_yscale('log')
        ax.set_xscale('log')

        
        ax.legend(loc='best')
        ax.set_aspect('equal', adjustable='box')
        fig.set_size_inches(7, 8, forward=True)

        plt.savefig(individual_save_path, dpi=1200)
        plt.close(fig)
                    
        
        # Plotting of 3 aggregated day forecast
        individual_save_path = join(validation_save_path, f'AggregatedValidationPlot.png')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rasterized = False
        ax.set(xlim=[0.5,200],ylim=[0.5,200])

        max_value = int(max(max(regular_arr), max(validation_arr)))
        min_value = int(min(min(regular_arr), min(validation_arr)))
        t = np.linspace(min_value, max_value, max_value)
        ax.plot(t, t, rasterized=rasterized, label='Optimal Fit',linewidth=2, color='red')
        ax.plot(regular_mean_arr, validation_mean_arr, '.', rasterized=rasterized, label='Average of 3 Withheld Days', color='black')

        title = 'Aggregated Validation Plot' 
        ax.set_title(title, fontdict = csfont)
        ax.set_ylabel('Validation Sample', fontdict = csfont)
        ax.set_xlabel('Regular Sample', fontdict=csfont)
        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.legend(loc='best')
        ax.set_aspect('equal', adjustable='box')

        fig.text(.5, .05, '(b)', ha='center', fontdict=csfont)
        fig.set_size_inches(7, 8, forward=True)
        plt.savefig(individual_save_path, dpi=1200)
        plt.close(fig)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='./data/', help='directory for the data')
    parser.add_argument('--results', default='./results', help='directory to save the results and plots in')
    parser.add_argument('--results-path', default=None, nargs='+', help='paths to the result folder to compare')
    args = parser.parse_args()

    model = ValidationResult(args)
import csv
import json
import argparse
import datetime as dt 
import pandas as pd 
import numpy as np
import scipy


from os.path import join, exists
from statsmodels.stats import weightstats as stests
from scipy import stats
from matplotlib import pyplot as plt

# plt.style.use('seaborn-darkgrid')
# plt.rcParams["font.family"] = "Helvetica"


class ValidationResult():
    """
        Compares two distributions via the Pearson Correlation Coefficient and saves the resulting score and p value in a csv file

        Usage:
            - specify 2 result directories to compare after the ---results_path argument
            -> the resulting 'comparison.csv' and 'final_comparison.csv' is saved in the first directory

        Example:
        python scripts/ValidationResult.py --results-path results/05_13_20_23_54_17_US_county_iter_200_warmup_100_num_counties_100_processing_Processing.REMOVE_NEGATIVE_VALUES/ results/05_14_20_01_27_52_US_county_validation_withholding_iter_200_warmup_100_num_counties_100_processing_Processing.REMOVE_NEGATIVE_VALUES/
    
        
    """
    def __init__(self, args):
        self.args = args
        for k, v in args.__dict__.items():
            setattr(self, k, v)

        self.summary_list = []
        for path in self.results_path:
            self.summary_list.append(self.parse_summary(path))
        

        # always save results into the first directory
        self.save_path = self.results_path[0]
        geocode_1 = self.get_geocode(self.results_path[0])
        geocode_2 = self.get_geocode(self.results_path[1])

        assert geocode_1 == geocode_2, ValueError('geocode.csv for the two paths not similar')

        self.geocode = geocode_1
        print(f'Length of geocode: {len(self.geocode)}')
        self.num_counties = self.get_num_counties(self.results_path[0])
        # print(f'Number of counties: {self.num_counties}')
        self.start_dates = self.get_start_dates(self.results_path[0])
        self.start_date, self.end_date, self.validation_days_list, self.validation_days_dict = self.get_validation_days(self.results_path[1])
        

        self.make_validation_plot(self.save_path, self.summary_list[0], self.summary_list[1], self.validation_days_list)

    def parse_summary(self, path):
        with open(join(path, 'summary.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)

            # read parameters
            # They are order in the summary file in the same way, i.e. from 1 to M
            mu = [] # length M
            kappa = [] # length 1 
            alpha = [] # length M 
            predicted_deaths = {} # M key-val pairs with length N
            rt_adj = {}  # M key-val pairs with length N
            
            self.parameter_name_list = ['mu', 'kappa', 'alpha', 'predicted_deaths', 'rt_adj']
            for row in reader:
                if row[0][:2] == 'mu':
                    mu.append(row[1:])
                
                if row[0][:5] == 'kappa':
                    kappa.append(row[1:])

                if row[0][:6] == 'alpha[':
                    alpha.append(row[1:])

                if row[0][:9] == 'E_deaths[':
                    pos = row[0][row[0].find('[')+1:row[0].find(']')]
                    t, m = pos.split(',')
                    predicted_deaths.setdefault(int(m), []).append(row[1:])
                
                if row[0][:6] == 'Rt_adj':
                    pos = row[0][row[0].find('[')+1:row[0].find(']')]
                    t, m = pos.split(',')
                    rt_adj.setdefault(int(m), []).append(row[1:])
                        

        return mu, kappa, alpha, predicted_deaths, rt_adj 



    def get_num_counties(self, path):
        with open(join(path, 'logfile.txt'),'r') as f:
            args = json.load(f)
        return args['M']   


    def get_start_dates(self, path):
        with open(join(path,'start_dates.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            start_dates = next(reader)[1:]
        return start_dates

    def get_geocode(self, path):
        with open(join(path,'geocode.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            geocode = next(reader)[1:]
        geocode = [i.zfill(5) for i in geocode]
        return geocode


    def get_validation_days(self, path):
        validation_days_dict = {}
        validation_days_list = []
        with open(join(path, 'validation_days.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            start_date, end_date = next(reader)
            next(reader)
            for row in reader:
                validation_days_dict[row[0]] = row[1:]
                validation_days_list.append([int(i) for i in row[1:]])


        return start_date, end_date, validation_days_list, validation_days_dict

                    


    def recursive_len(self, item):
        if type(item) == list:
            return sum(self.recursive_len(subitem) for subitem in item)
        else:
            return 1
    
    def make_validation_plot(self, path, summary1, summary2, validation_days_list):
        validation_save_path = join(path, 'plots', 'validation')
        
        if not exists(validation_save_path):
            os.mkdir(validation_save_path)

        validation_days_list_length  = self.recursive_len(validation_days_list) #sum([len(i) for i in validation_days_list])
        
        regular_list = []
        validation_list = []
        regular_mean_list = []
        validation_mean_list = []
        for i in range(len(summary1)-1):
            if  isinstance(summary1[i], dict) and 'death' in self.parameter_name_list[i]:
                assert (len(validation_days_list) == len(summary1[i])), f'Validation days have length {len(validation_days_list)} and summary: {len(summary1[i])}'
                for idx, ((key1, val1), (key2, val2)) in enumerate(zip(summary1[i].items(), summary2[i].items())): # length number counties
                    assert (len(val1) == len(val2)), f'summaries have different lengths'
                    validation_days_current = validation_days_list[idx]
                    regular_mean_list.append([])
                    validation_mean_list.append([])
                    for value_list in [val1, val2]:
                        for j in range(len(value_list)):
                            adjusted_date = dt.datetime.strptime(self.start_dates[idx],'%m/%d/%y').toordinal() - dt.datetime.strptime(self.start_date, '%m/%d/%y').toordinal() + j 
                            # print(f'county number: {idx} || date {adjusted_date}')
                            if adjusted_date in validation_days_current:
                                #print(idx,j,adjusted_date)
                                if value_list == val1:
                                    regular_list.append(value_list[j])
                                    regular_mean_list[idx].append(value_list[j][0]) # only take the mean (first value)
                                else:
                                    validation_list.append(value_list[j])
                                    validation_mean_list[idx].append(value_list[j][0]) # only take the mean (first value)
                    
                    regular_mean_list[idx] = np.mean(np.array(regular_mean_list[idx]).astype(np.float))
                    validation_mean_list[idx] = np.mean(np.array(validation_mean_list[idx]).astype(np.float))

        print(f'Expected length of validation_days_list: {validation_days_list_length} \nActual length of regular list: {len(regular_list)}') 
        print(f'Expected length of validation_days_list: {validation_days_list_length} \nActual length of validation list: {len(validation_list)}') 
        

        # Prepare arrays 
        regular_arr = np.array(regular_list).astype(np.float)[:,0]
        validation_arr = np.array(validation_list).astype(np.float)[:,0]
        regular_mean_arr = np.array(regular_mean_list)
        validation_mean_arr = np.array(validation_mean_list)
        
        idx_reg = np.argwhere(np.isnan(regular_mean_arr))
        idx_val = np.argwhere(np.isnan(validation_mean_arr))
        
        assert idx_reg.all() == idx_val.all()   
        
        regular_mean_arr = regular_mean_arr[~np.isnan(regular_mean_arr)]
        validation_mean_arr = validation_mean_arr[~np.isnan(validation_mean_arr)]
        # similarity measures
        pearson, pval = stats.pearsonr(regular_arr, validation_arr)
        pearson_aggregated, pval_aggregated = stats.pearsonr(regular_mean_arr, validation_mean_arr)
        
        l1 = np.linalg.norm((regular_arr - validation_arr), ord=1)
        l1_aggregated = np.linalg.norm((regular_mean_arr - validation_mean_arr), ord=1)
        print(f'L1: {l1}|| L1_agg: {l1_aggregated}')
        print(f'R: {pearson}|| R_agg: {pearson_aggregated}')
        

        with open(join(self.save_path,'pearson.csv'), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([f'Length of non aggregated array {len(regular_arr)}']) 
            writer.writerow([f'Length of  aggregated array {len(regular_mean_arr)}']) 
            writer.writerow(['Pearson R value regular', 'Pearson R value aggregated'])
            writer.writerow([pearson, pearson_aggregated])
            writer.writerow(['pvalues'])
            writer.writerow([pval, pval_aggregated])
            writer.writerow(['regular', 'validation', 'regular_aggregated', 'validation aggregated'])
            
            for i in range(len(regular_arr)):
                if i < len(validation_mean_arr):
                    arr = [regular_arr[i], validation_arr[i], regular_mean_arr[i], validation_mean_arr[i]]
                else: 
                    arr = [regular_arr[i], validation_arr[i]]
                writer.writerow(arr)



        # Plotting of 3 day forecast
        individual_save_path = join(validation_save_path, f'ValidationPlot.png')
        

        fig = plt.figure()
        csfont = {'fontsize':20}
        #fig.add_axes()
        ax = fig.add_subplot(111)
        #ax2 = fig.add_subplot(122)
        rasterized = False
        ax.set(xlim=[0.5,500],ylim=[0.5,500])

        max_value = int(max(max(regular_arr), max(validation_arr))) + 1000
        min_value = int(min(min(regular_arr), min(validation_arr)))
        t = np.linspace(min_value, max_value, max_value)
        ax.plot(t, t, rasterized=rasterized, label='Optimal Fit',linewidth=2, color='red')
        ax.plot(regular_arr, validation_arr, '.', rasterized=rasterized, label='Withheld Days', color='black')

        title = 'Validation Plot'
        ax.set_title(title, fontdict = csfont)
        ax.set_ylabel('Validation Sample', fontdict = csfont)
        ax.set_xlabel('Regular Sample', fontdict=csfont)
        fig.text(.5, .05, '(a)', ha='center', fontdict=csfont)
        ax.set_yscale('log')
        ax.set_xscale('log')

        
        ax.legend(loc='best')
        ax.set_aspect('equal', adjustable='box')
        fig.set_size_inches(7, 8, forward=True)

        plt.savefig(individual_save_path, dpi=1200)
        plt.close(fig)
                    
        
        # Plotting of 3 aggregated day forecast
        individual_save_path = join(validation_save_path, f'AggregatedValidationPlot.png')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rasterized = False
        ax.set(xlim=[0.5,200],ylim=[0.5,200])

        max_value = int(max(max(regular_arr), max(validation_arr)))
        min_value = int(min(min(regular_arr), min(validation_arr)))
        t = np.linspace(min_value, max_value, max_value)
        ax.plot(t, t, rasterized=rasterized, label='Optimal Fit',linewidth=2, color='red')
        ax.plot(regular_mean_arr, validation_mean_arr, '.', rasterized=rasterized, label='Average of 3 Withheld Days', color='black')

        title = 'Aggregated Validation Plot' 
        ax.set_title(title, fontdict = csfont)
        ax.set_ylabel('Validation Sample', fontdict = csfont)
        ax.set_xlabel('Regular Sample', fontdict=csfont)
        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.legend(loc='best')
        ax.set_aspect('equal', adjustable='box')

        fig.text(.5, .05, '(b)', ha='center', fontdict=csfont)
        fig.set_size_inches(7, 8, forward=True)
        plt.savefig(individual_save_path, dpi=1200)
        plt.close(fig)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='./data/', help='directory for the data')
    parser.add_argument('--results', default='./results', help='directory to save the results and plots in')
    parser.add_argument('--results-path', default=None, nargs='+', help='paths to the result folder to compare')
    args = parser.parse_args()

    model = ValidationResult(args)
