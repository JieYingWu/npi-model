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
        Compares two distributions via the ks-test and saves the resulting score and p value in a csv file

        Usage:
            - specify 2 result directories to compare after the ---results_path argument
            -> the resulting 'comparison.csv' is saved in the first directory

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
        self.start_dates = self.get_start_dates(self.results_path[0])
        self.start_date, self.end_date, self.validation_days_list, self.validation_days_dict = self.get_validation_days(self.results_path[1])
        print(self.validation_days_list)
        # final_dict, final_list = self.compare_2_daily(self.summary_list[0], self.summary_list[1])
        final_dict, final_list = self.compare_2(self.summary_list[0], self.summary_list[1])
        

        if self.plot_val:
            self.make_validation_plot(self.save_path, self.summary_list[0], self.summary_list[1], self.validation_days_list)
        # write results to a csv 
        self.write_results(self.save_path, final_list)
        self.write_final_results(final_list, self.num_counties)

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

                    
    def compare_2_daily(self, summary1, summary2):
        final_dict = {}
        final_list = []
        
        if self.test == 'z':
            test_function = self.z_test
        elif self.test == 'ks':
            test_function = self.ks_test

        for i in range(len(summary1)):
            if  isinstance(summary1[i], list):
                for j in range(len(summary1[i])):
                    print(self.parameter_name_list[i],j)
                    identifier = '_'.join([self.parameter_name_list[i], str(j)])
                    final_dict[identifier] = test_function(summary1[i][j], summary2[i][j])
                    final_list.append([identifier]+list(test_function(summary1[i][j], summary2[i][j])))
            else:
                for (key1, val1), (key2, val2) in zip(summary1[i].items(), summary2[i].items()):
                    for k in range(len(val1)):
                        print(self.parameter_name_list[i], key1, k)
                        identifier = '_'.join([self.parameter_name_list[i], str(key1), str(k)])
                        final_list.append([identifier]+list(test_function(val1[k],val2[k])))
                        final_dict[identifier] = test_function(val1[k], val2[k])

        return final_dict, final_list
    
    

    def compare_2(self, summary1, summary2):
        final_list = []
        final_dict = {}
        
        test_function = self.ks_test

        for i in range(len(summary1)):
            if  isinstance(summary1[i], dict):
                for (key1, val1), (key2, val2) in zip(summary1[i].items(), summary2[i].items()):
                    identifier = '_'.join([self.parameter_name_list[i], str(key1)])

                
                    
                    # pick the mean
                    try:
                        dis_1 = np.array(val1)[:,0].astype(np.float)
                        dis_2 = np.array(val2)[:,0].astype(np.float)
                    except ValueError:
                        print(dis_1)
                        print(dis_2)
                    statistic, pval = stats.ks_2samp(dis_1, dis_2)
                    final_dict[identifier] = (statistic, pval)
                    final_list.append([identifier]+[statistic, pval])
                    if self.plot_qq:
                        self.make_qq_plots(self.save_path, dis_1, dis_2, self.geocode[key1-1], statistic, pval, self.parameter_name_list[i], identifier)
                    if self.plot_dis:
                        self.make_distribution_plots(self.save_path, dis_1, dis_2, self.geocode[key1-1], statistic, pval, self.parameter_name_list[i])
                    
        return final_dict, final_list
    



    def z_test(self, distribution1, distribution2):
        """ distribution is a tuple:
        - mean
        - standard error mean
        - std
        - 2.5%
        - 25%
        - 50 %
        - 75 %
        - 97.5%
        - n_eff
        - R_hat
        """
        delta = 0
        mean_1 = float(distribution1[0])
        mean_2 = float(distribution2[0])
        std_1 = float(distribution1[2])
        std_2 = float(distribution2[2])
        se_1 = float(distribution1[1])
        se_2 = float(distribution2[1])



        pooledSE = np.sqrt(se_1**2 + se_2**2)
        z = ((mean_1 - mean_2) - delta)/pooledSE
        pval = 2*(1 - stats.norm.cdf(np.abs(z)))
        return np.round(z, 3), np.round(pval, 4)



    def ks_test(self, distribution1, distribution2):
        """ distribution is a tuple:
            - mean
            - standard error mean
            - std
            - 2.5%
            - 25%
            - 50 %
            - 75 %
            - 97.5%
            - n_eff
            - R_hat"""
        # Assume normal distributions for the distributions
        np.random.seed(0)
        x = np.random.normal(float(distribution1[0]), float(distribution1[2]), 10000)
        y = np.random.normal(float(distribution2[0]), float(distribution2[2]), 10000)

        statistic, pvalue = scipy.stats.ks_2samp(x,y)

        return statistic, pvalue

    def write_results(self, path, final_list):
        test_value = self.test
        with open(join(path,'comparison.csv'), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')

            writer.writerow([f'Comparing {self.results_path[0]} and {self.results_path[1]}'])
            writer.writerow(['-----------------------------'])
            writer.writerow(['identifier', f'{self.test}','p-val'])
            writer.writerows(final_list)

    def make_qq_plots(self, path, timeseries_1, timeseries_2, fips, statistic, pval, tag, identifier):
        save_path = join(path, 'plots','qq')
        print(f'Plotting {identifier}')
        if fips == '36061':
            print(timeseries_1)
            print(timeseries_2)
        if not exists(save_path):
            os.mkdir(save_path)

        # storage efficient plotting
        rasterized = False
        
        max_value = int(max(max(timeseries_1), max(timeseries_2)))+1
        min_value = int(min(min(timeseries_1), min(timeseries_2)))
        individual_save_path = join(save_path, f'{tag}_{fips}_qq_plot.png')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        #Set Font 
        csfont = {'fontname':'Times New Roman'}

        # ax.set(xlim=[0,1],ylim=[0,1])
        # ax.scatter(np.linspace(0, len(timeseries_1), len(timeseries_1)), timeseries_1)


        t = np.linspace(min_value, max_value, max_value)
        ax.plot(t,t, rasterized=rasterized, label='Optimal Fit',linewidth=2, color='orange')
        ax.plot(sorted(timeseries_1), sorted(timeseries_2), '.', rasterized=rasterized, label='Timeseries', color='blue')
        title = 'Q-Q Plot for County:{} for {} \n {} score: {:.2f}, pvalue:{:.2f}'.format(fips, tag, self.test, statistic, pval)
        ax.set_title(title, fontdict = csfont)
        ax.set_ylabel('Validation Sample', fontdict = csfont)
        ax.set_xlabel('Regular Sample', fontdict=csfont)
        ax.legend(loc='best')
        plt.savefig(individual_save_path, dpi=100)
        plt.close(fig)


    def make_distribution_plots(self, path, timeseries_1, timeseries_2, fips, statistic, pval, tag):
        distribution_save_path = join(path, 'plots', 'mean_expected_deaths')

        if not exists(distribution_save_path):
            os.mkdir(distribution_save_path)

        individual_save_path = join(distribution_save_path, f'{tag}_{fips}_mean_plot.png')
        fig = plt.figure()
        csfont = {'fontname':'Times New Roman'}
        ax = fig.add_subplot(111)
        # ax.set(xlim=[0,150],ylim=[0,])
        ax.scatter(np.linspace(0, len(timeseries_1), len(timeseries_1)), timeseries_1, label='Sample 1')
        ax.scatter(np.linspace(0, len(timeseries_2), len(timeseries_2)), timeseries_2, label='Sample 2')


        title = 'Distributions Plot for County:{} for {} \n {} score: {:.2f}, pvalue:{:.2f}'.format(fips, tag, self.test, statistic, pval)
        ax.set_title(title, fontdict = csfont)
        ax.set_ylabel('Daily Deaths', fontdict = csfont)
        ax.set_xlabel('Days', fontdict=csfont)
        ax.legend(loc='best')
        plt.savefig(individual_save_path, dpi=100)
        plt.close(fig)


    def make_validation_plot(self, path, summary1, summary2, validation_days_list):
        validation_save_path = join(path, 'plots', 'validation')
        
        if not exists(validation_save_path):
            os.mkdir(validation_save_path)

        validation_days_list_length  = sum([len(i) for i in validation_days_list])
        
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
        print(regular_arr)
        print(validation_arr)
        regular_mean_arr = np.array(regular_mean_list)
        validation_mean_arr = np.array(validation_mean_list)
        
        idx_reg = np.argwhere(np.isnan(regular_mean_arr))
        idx_val = np.argwhere(np.isnan(validation_mean_arr))
        
        assert idx_reg.all() == idx_val.all()   
        
        regular_mean_arr = regular_mean_arr[~np.isnan(regular_mean_arr)]
        validation_mean_arr = validation_mean_arr[~np.isnan(validation_mean_arr)]
        # similarity measures
        pearson, _ = stats.pearsonr(regular_arr, validation_arr)
        pearson_aggregated, _ = stats.pearsonr(regular_mean_arr, validation_mean_arr)
        
        l1 = np.linalg.norm((regular_arr - validation_arr), ord=1)
        l1_aggregated = np.linalg.norm((regular_mean_arr - validation_mean_arr), ord=1)
        print(f'L1: {l1}|| L1_agg: {l1_aggregated}')
        print(f'R: {pearson}|| R_agg: {pearson_aggregated}')
        

        with open('pearson.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['Pearson R value regular', 'Pearson R value aggregated'])

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
        

    def write_final_results(self, final_list, num_counties):
        # select only the first num counties entries which correspont to the predicted deaths
        final_list = final_list[:num_counties]
        pval_arr = np.array(final_list)[:,2].astype(np.float)
        statistic_arr = np.array(final_list)[:,1].astype(np.float)

        max_pvalue = max(pval_arr)
        argmax_pvalue = self.geocode[np.argmax(pval_arr)]
        min_pvalue = min(pval_arr)
        argmin_pvalue = self.geocode[np.argmin(pval_arr)]
        qtl_95_pvalue = np.percentile(pval_arr, 95)
        qtl_5_pvalue = np.percentile(pval_arr, 5)

        mean_pvalue = np.mean(pval_arr)
        std_pvalue = np.std(pval_arr)
        median_pvalue = np.median(pval_arr)

        max_statistic = max(statistic_arr)
        argmax_statistic = self.geocode[np.argmax(statistic_arr)]
        min_statistic = min(statistic_arr)
        argmin_statistic = self.geocode[np.argmin(statistic_arr)]
        qtl_95_statistic = np.percentile(statistic_arr, 95)
        qtl_5_statistic = np.percentile(statistic_arr, 5)

        mean_statistic = np.mean(statistic_arr)
        std_statistic = np.std(statistic_arr)
        median_statistic = np.median(statistic_arr)

        values_list = [max_pvalue, argmax_pvalue, min_pvalue, argmin_pvalue, mean_pvalue, std_pvalue, median_pvalue, qtl_95_pvalue, qtl_5_pvalue, max_statistic, argmax_statistic, min_statistic, argmin_statistic, mean_statistic, std_statistic, median_statistic, qtl_95_statistic, qtl_5_statistic]

        values_names_list = ['max_pvalue', 'FIPS_max_pvalue', 'min_pvalue', 'FIPS_min_pvalue', 'mean_pvalue', 'std_pvalue', 'median_pvalue','95_percentile_pvalue','5_percentile_pvalue', 'max_statistic', 'FIPS_max_statistic', 'min_statistic', 'FIPS_min_statistic', 'mean_statistic', 'std_statistic', 'median_statistic','95_percentile_statistic','5_percentile_statistic']
        
        list_to_write = [[name, value] for name, value in zip(values_names_list, values_list)]

        with open(join(self.save_path, 'final_comparison.csv'),'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(list_to_write)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='./data/', help='directory for the data')
    parser.add_argument('--results', default='./results', help='directory to save the results and plots in')
    parser.add_argument('--results-path', default=None, nargs='+', help='paths to the result folder to compare')
    parser.add_argument('--test', default='ks', choices=['ks', 'z'], help='test to compare distributions')
    parser.add_argument('--plot-dis', action='store_true', help='whether to plot the distribution')
    parser.add_argument('--plot-val', action='store_true', help='whether to make the validation  plot')
    parser.add_argument('--plot-qq', action='store_true', help='whether to make the qq plots')
    #parser.add_argument('-val', choices=[1, 2, 3], help='Types of validation to compare')
    args = parser.parse_args()

    model = ValidationResult(args)
