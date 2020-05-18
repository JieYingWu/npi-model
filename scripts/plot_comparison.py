import os
import json 
import argparse 
import csv 
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists



def parse_csv(path):    
    with open(join(path,'comparison_z.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        next(reader)
        header = next(reader)
        
        test = header[1]
        mu = []
        alpha = []
        deaths = []

        for row in reader:
            if 'mu' in row[0]:
                mu.append(row[1:])
            if 'alpha' in row[0]:
                alpha.append(row[1:])

            if 'deaths' in row[0]:
                deaths.append(row[1:])

    return mu, alpha, deaths, test    


def get_num_counties(path):
    with open(join(path, 'logfile.txt'),'r') as f:
        args = json.load(f)
    print(args)
    return args['M']   


def get_start_dates(path):
    with open(join(path,'start_dates.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        start_dates = next(reader)[1:]
    return start_dates

def get_geocode(path):
    with open(join(path,'geocode.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        geocode = next(reader)
    return geocode


def plot_pval_deaths_county(path, deaths, county, test):
    """ which county to plot
    param deaths: list of tuples( statistic, pvalue)
    param county: number of county to plot (0- num counties)
    """
    num_counties = get_num_counties(path)
    start_dates = get_start_dates(path)
    geocode = get_geocode(path)
    print(len(start_dates))
    assert (0 <= county < num_counties), ValueError


    # slice the deaths according to the requested county
    deaths_to_plot = deaths[county*num_counties:(county+1)*num_counties]
    # select only the p value to plot 
    deaths_to_plot = [j for (i,j) in deaths_to_plot]
    deaths_arr = np.array(deaths_to_plot, dtype=np.float)
    

    # get the number of days to plot 
    first_date = dt.datetime.strptime(start_dates[county], '%m/%d/%y').toordinal()
    last_date = dt.datetime.strptime('5/14/20', '%m/%d/%y').toordinal()
    print(deaths_arr)

    print(last_date-first_date)
    t = np.linspace(0, last_date-first_date, num=last_date-first_date)
    print(t)
    deaths_arr = deaths_arr[:last_date-first_date]

    assert len(t) == len(deaths_arr), f'Check length of dates({len(t)})/ deaths array({len(deaths_arr)})'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(t, deaths_arr)
    ax.set(title=f'Pvalue for County {geocode[county]}',
            ylabel='p-value',
            xlabel='Days')
    if not exists(join(path, 'plots',f'comparison_{test}-test' )):
        os.mkdir(join(path, 'plots',f'comparison_{test}-test'))
    save_path = join(path, 'plots',f'comparison_{test}-test', f'{geocode[county]}_pvalue_plot.png')
    plt.savefig(save_path)
    plt.show()




    pass 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', default=None, help='Path to unique results folder with comparison.csv inside of it')
    parser.add_argument('--county', type=int, help='Integer in range (0, length(number_counties)))')
    args = parser.parse_args()

    mu, alpha ,deaths, test = parse_csv(args.path)
    M = get_num_counties(args.path)
    if not args.county:
        for i in range(M):
            plot_pval_deaths_county(args.path, deaths, i, test)
    else:
        plot_pval_deaths_county(args.path, deaths, args.county, test)
