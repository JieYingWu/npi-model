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

plt.style.use('seaborn-darkgrid')


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
        print(self.results_path[0])
        geocode_1 = self.get_geocode(self.results_path[0])
        geocode_2 = self.get_geocode(self.results_path[1])

        assert geocode_1 == geocode_2, ValueError('geocode.csv for the two paths not similar')

        self.geocode = geocode_1
        self.num_counties = self.get_num_counties(self.results_path[0])
        self.start_date = self.get_start_dates(self.results_path[0])

        # final_dict, final_list = self.compare_2_daily(self.summary_list[0], self.summary_list[1])
        final_dict, final_list = self.compare_2(self.summary_list[0], self.summary_list[1])
        
        # write results to a csv 
        self.write_results(self.save_path, final_list)


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

                if row[0][:9] == 'E_deaths0':
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

    def get_geocode(self,path):
        with open(join(path,'geocode.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            geocode = next(reader)
        geocode = [i.zfill(5) for i in geocode]
        return geocode

                    
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
                print(len(summary1[i]))
                for (key1, val1), (key2, val2) in zip(summary1[i].items(), summary2[i].items()):
                    identifier = '_'.join([self.parameter_name_list[i], str(key1)])
                    print(identifier)
                    # pick the mean
                    dis_1 = np.array(val1)[:,0].astype(np.float)
                    dis_2 = np.array(val2)[:,0].astype(np.float)
                    
                    statistic, pval = stats.ks_2samp(dis_1, dis_2)

                    final_dict[identifier] = (statistic, pval)
                    final_list.append([identifier]+[statistic, pval])

                    self.plot_qq(self.save_path, dis_1, dis_2, self.geocode[key1], statistic, pval, self.parameter_name_list[i])
        
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
            print(type(final_list))
            writer.writerows(final_list)

    def plot_qq(self, path, timeseries_1, timeseries_2, fips, statistic, pval, tag):
        save_path = join(path, 'plots','qq')

        if not exists(save_path):
            os.mkdir(save_path)
        # storage efficient plotting
        rasterized = False
        
        max_value = int(max(max(timeseries_1), max(timeseries_2)))
        min_value = int(min(min(timeseries_1), min(timeseries_2)))
        individual_save_path = join(save_path, f'{tag}_{fips}_qq_plot.pdf')
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # ax.set(xlim=[0,1],ylim=[0,1])
        # ax.scatter(np.linspace(0, len(timeseries_1), len(timeseries_1)), timeseries_1)
        ax.plot(sorted(timeseries_1), sorted(timeseries_2), '.', rasterized=rasterized, label='Timeseries')


        t = np.linspace(min_value, max_value, max_value)
        ax.plot(t,t, rasterized=rasterized, label='Perfect Fit')
        title = 'Q-Q Plot for County:{} for {} \n {} score: {:.2f}, pvalue:{:.2f}'.format(fips, tag, self.test, statistic, pval)
        ax.set(title=title,
                ylabel='Validation Sample',
                xlabel='Regular Sample')
        ax.legend(loc='best')
        plt.savefig(individual_save_path)

        plt.close(fig)

    def write_final_results(self, final_list, num_counties):
        # select only the first num counties entries which correspont to the predicted deaths
        final_list = final_list[:num_counties]
        final_arr = np.array(final_list)
        max_value = max(final_arr)
        min_value = min(final_arr)

        mean = np.mean(final_arr)
        std = np.std(final_arr)
        median = np.median(final_arr)

        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='./data/', help='directory for the data')
    parser.add_argument('--results', default='./results', help='directory to save the results and plots in')
    parser.add_argument('--results-path', default=None, nargs='+', help='paths to the result folder to compare')
    parser.add_argument('--test', default='ks', choices=['ks', 'z'], help='test to compare distributions')
    #parser.add_argument('-val', choices=[1, 2, 3], help='Types of validation to compare')
    args = parser.parse_args()

    model = ValidationResult(args)
