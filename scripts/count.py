from os.path import join, exists
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from data_parser import impute, remove_negative_values

def read_timeseries(fips, end_date):
    cases = pd.read_csv("data/us_data/infections_timeseries_w_states.csv", index_col=0, delimiter=',')
    deaths = pd.read_csv("data/us_data/deaths_timeseries_w_states.csv", index_col=0, delimiter=',')
    total_cases = np.zeros(len(fips))
    total_deaths = np.zeros(len(fips))
    for i in range(len(fips)):
        f = fips[i]
        total_cases[i] = cases.loc[f,end_date]
        total_deaths[i] = deaths.loc[f,end_date]
    return total_cases, total_deaths

def count(region, end_date):
    dict_of_start_dates = pd.read_csv('results/' + region + '_start_dates.csv', delimiter=',', index_col=0)
    start_dates = dict_of_start_dates.values.tolist()[0]
    dict_of_fips = pd.read_csv('results/' + region + '_geocode.csv', delimiter=',', index_col=0)
    list_of_fips = dict_of_fips.values.tolist()[0]

    total_cases, total_deaths = read_timeseries(list_of_fips, end_date)
    predict_cases = np.zeros(len(list_of_fips))
    predict_deaths = np.zeros(len(list_of_fips))
    df = pd.read_csv('results/' + region + '_summary.csv', delimiter=',', index_col=0)
    for i in range(len(start_dates)):
        fips = list_of_fips[i]
        base = datetime.datetime.strptime(str(start_dates[i]), '%m/%d/%y')
        days_to_add = (datetime.datetime.strptime(end_date, '%m/%d/%y') - base).days

        predict_case = 0
        predict_death = 0
        region_number = str(i + 1) + ']'
        for j in range(days_to_add):
            case_name = 'prediction[' +  str(j+1) + ',' + region_number
            death_name = 'E_deaths[' +  str(j+1) + ',' + region_number
            predict_case += df.loc[case_name, 'mean']
            predict_death += df.loc[death_name, 'mean']

        
        predict_cases[i] = predict_case
        predict_deaths[i] = predict_death

    write_out = np.stack((list_of_fips, total_cases, predict_cases, total_deaths, predict_deaths), axis=1)
    np.savetxt('results/counts.csv', write_out, delimiter=',')
        
def main():
    region = sys.argv[1]
    end_date = '4/27/20'  # Count to this date
    count(region, end_date)
    
if __name__ == '__main__':
    main()
