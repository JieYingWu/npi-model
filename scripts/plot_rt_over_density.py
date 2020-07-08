import os
from os.path import join, exists
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import json
import seaborn as sns
import math


def get_means_list(path, geo_list):
    # 1 for deaths; 0 for infections
    plot_name = "mu"
    df = pd.read_csv(path + "/summary.csv", delimiter=',', index_col=0)
    row_names = list(df.index.tolist())
    means = {}
    for num_of_country in range(0, len(geo_list)):
        county_number = "[" + str(int(num_of_country) + 1) + "]"
        for name in row_names:
            if plot_name in name and county_number in name:
                rowData = df.loc[name, :]
                key = geo_list[num_of_country]
                means[key] = rowData['mean']
    return means


def create_geocodes_dict(path_rt, path_density):
    df = pd.read_csv(path_rt + "/geocode.csv", delimiter=',', index_col=0)
    supercounties_names = list(df.loc[0][:])

    with open(path_density + "/supercounties.json") as f:
        super_file = json.load(f)

    selected_dict = {}
    i = 0
    for name in supercounties_names:
        if "_" in str(name):
            selected_dict[name] = super_file[name]
        else:
            name = str(name).zfill(5)
            selected_dict[name] = [name]
            supercounties_names[i] = name
        i += 1
    return selected_dict, supercounties_names


def create_deaths_dict(path_density, date, selected_dict):
    df_deaths = pd.read_csv(path_density + "/deaths_timeseries_w_states.csv", delimiter=',')
    df_deaths = df_deaths.set_index('FIPS')
    dict_deaths = {}

    for key in selected_dict.keys():
        for county in selected_dict[key]:
       
            dict_deaths[county] = df_deaths.at[int(county), date]
    print(dict_deaths)
    return dict_deaths


def read_density_sum(path_density, selected_dict, r0_means_list, plot_variable, pos, start_day_dict):
    df = pd.read_csv(path_density + "/counties.csv", delimiter=',')
    df = df.set_index('FIPS')
    sum_density_dict = {}
    r0_dict = {}
    # pass the date for r0 plots
    # division over the zero
    df_deaths = pd.read_csv(path_density + "/deaths_timeseries_w_states.csv", delimiter=',')
    df_deaths = df_deaths.set_index('FIPS')
    for key in selected_dict.keys():
        pop_sum = 0
        total = 0
        if use_death_weight:
            for county in selected_dict[key]:
                if pos == 0:
                    date = start_day_dict[key]
                else:
                    date = dates[pos-1]
                total = total + df_deaths.at[int(county), date]
            for county in selected_dict[key]:
                if pos == 0:
                    date = start_day_dict[key]
                else:
                    date = dates[pos-1]
                weighted_avg = (df_deaths.at[int(county), date] / total)
                pop_sum = pop_sum + df.at[int(county), plot_variable] * weighted_avg

        elif use_weight_average:
            for county in selected_dict[key]:
                total = total + df.at[int(county), 'POP_ESTIMATE_2018']
            for county in selected_dict[key]:
                weighted_avg = (df.at[int(county), 'POP_ESTIMATE_2018'] / total)
                pop_sum = pop_sum + df.at[int(county), plot_variable] * weighted_avg
        else:
            for county in selected_dict[key]:
                pop_sum = pop_sum + df.at[int(county), plot_variable]

        sum_density_dict[key] = pop_sum
        r0_dict[key] = r0_means_list[key]
    return sum_density_dict, r0_dict


def read_density(path_density, selected_dict, dict_r0_supercounty, plot_variable):
    df = pd.read_csv(path_density + "/counties.csv", delimiter=',')
    df = df.set_index('FIPS')
    density_dict = {}
    dict_r0 = {}
    for key in selected_dict.keys():
        for county in selected_dict[key]:
            density_dict[county] = df.at[int(county), plot_variable]  # / df.at[int(county),'POP_ESTIMATE_2018']
            dict_r0[county] = dict_r0_supercounty[key]
    return density_dict, dict_r0


def get_rt_adj(path, geo_list, start_day_dict):
    # 1 for deaths; 0 for infections
    plot_name = "Rt_adj["
    df = pd.read_csv(path + "/summary.csv", delimiter=',', index_col=0)
    row_names = list(df.index.tolist())
    means = {}
    for num_of_country in range(0, len(geo_list)):
        county_number = str(int(num_of_country) + 1) + "]"
        for name in row_names:
            if plot_name in name and name.split(",")[1] == county_number:
                rowData = df.loc[name, :]
                key = geo_list[num_of_country]
                if start_day_dict[key] > 0:  # in case there is first few plots will miss some counties that haven't begun yet.
                    if name.split(",")[0] == (plot_name + str(start_day_dict[key])):
                        means[key] = rowData['mean']
                else:
                    means[key] = None
    return means


def get_start_day_dict(path, geo_list, date_plot):
    dict_start_days = {}
    df = pd.read_csv(path + "/start_dates.csv", delimiter=',', index_col=0)
    supercounties_dates = list(df.loc[0][:])
    set_day_for_search = datetime.datetime.strptime(date_plot, '%m/%d/%y')
    for i in range(0, len(geo_list)):
        days_to_predict = (set_day_for_search - datetime.datetime.strptime(supercounties_dates[i], '%m/%d/%y')).days
        dict_start_days[geo_list[i]] = days_to_predict
    return dict_start_days


# get the start day of infection for each geo
def get_start_day(path, geo_list):
    dict_start_days = {}
    df = pd.read_csv(path + "/start_dates.csv", delimiter=',', index_col=0)
    supercounties_dates = list(df.loc[0][:])
    for i in range(0, len(geo_list)):
        dict_start_days[geo_list[i]] = supercounties_dates[i]
    return dict_start_days


# plot the R0
def plot_scatter_r0(path, plot_variable):
    pos = 0
    path_density = "../data/us_data"
    ax[pos].set_title("R0", pad=15)
    colors = ['#D55E00', '#CC79A7','#0072B2','#F0E442','#009E73']
    x_array = []

    for cluster_n in range(0, 5):
        print(cluster_n)
        path_rt = path + str(cluster_n)
        supercounties_dist, supercounties_names = create_geocodes_dict(path_rt, path_density)

        start_day_dict = get_start_day(path_rt, supercounties_names)
        r0_means_list = get_means_list(path_rt, supercounties_names)
        # to plot all supercouties only
        if plot_supercounties:
            density_dict, dict_r0 = read_density_sum(path_density, supercounties_dist, r0_means_list, plot_variable, pos, start_day_dict)
        # to plot all counties
        else:
            density_dict, dict_r0 = read_density(path_density, supercounties_dist, r0_means_list, plot_variable)

        y_list = []
        x_list = []
        for key in density_dict.keys():
            x = density_dict[key]
            y = dict_r0[key]
            if (y is not None) and (not math.isnan(x)):
                y_list.append(y)
                x_list.append(x) # since the X density will be the same, save it
            ax[pos].scatter(x, y, color=colors[cluster_n], s=8, alpha=set_transparency)

        # plot distrubution over Y axis
        ax2 = ax[pos].twiny()
        sns.distplot(y_list, hist=False, kde=True, vertical=True, norm_hist=True,
                     bins=10, color=colors[cluster_n],
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'shade': True, 'linewidth': 1},
                     ax=ax2)
        ax2.tick_params(axis='x')
        ax2.set_xlim(0, 10)
        ax2.tick_params(labeltop=False)

        # plot the histogram over the X axis
        ax3 = ax[pos].twinx()
        sns.distplot(x_list, hist=False, kde=True,  norm_hist=True,
                     bins=15, color=colors[cluster_n],
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'shade': True, 'linewidth': 1},
                     ax=ax3)

        # Making the labels pretty
        ax3.tick_params(axis='y')
        ax3.tick_params(labelright=False)
        ax3.tick_params(labeltop=False)
        ax[pos].tick_params(axis='x', labelrotation=45)
        ax[pos].set_ylabel('Reproductive Ratio')
        x_array.append(x_list)

        # set different ranges for different features
        if idx == 0:
            ax3.set_ylim(0, 1e-8)  # transit
            ax[pos].set_xlim(0, 1e10) # transit
        if idx == 1:
            ax3.set_ylim(0, 5e-4) # income
            ax[pos].set_xlim(0, 1e5)  # income
        if idx == 2:
            ax3.set_ylim(0, 6e-2) # density
            ax[pos].set_xlim(0, 2500) # density
    return x_array


# plot the Rt over the dates
def plot_scatter_radj(path, date_plot, pos, plot_variable,x_array):
    path_density = "../data/us_data"
    ax[pos].set_title(date_plot, pad=17)
    colors = ['#D55E00', '#CC79A7','#0072B2','#F0E442','#009E73']

    for cluster_n in range(0, 5):
        print("Retrieving information for cluster number ...", cluster_n)
        path_rt = path + str(cluster_n)
        supercounties_dist, supercounties_names = create_geocodes_dict(path_rt, path_density)

        start_day_dict = get_start_day_dict(path_rt, supercounties_names, date_plot)
        rt_adj_list = get_rt_adj(path_rt, supercounties_names, start_day_dict)

        # to plot all supercouties only
        if plot_supercounties:
            density_dict, dict_r0 = read_density_sum(path_density, supercounties_dist, rt_adj_list, plot_variable, pos, start_day_dict)
        # to plot all counties
        else:
            density_dict, dict_r0 = read_density(path_density, supercounties_dist, rt_adj_list, plot_variable)

        # creating a scatter plot
        y_list = []
        for key in density_dict.keys():
            x = density_dict[key]
            y = dict_r0[key]
            if (y is not None) and (not math.isnan(x)):
                y_list.append(y)
            ax[pos].scatter(x, y, color=colors[cluster_n], s=8, alpha=set_transparency)

        # creating a density plot on Y axis
        ax2 = ax[pos].twiny()
        sns.distplot(y_list, hist=False, kde=True, vertical=True, norm_hist=True,
                     bins=10, color=colors[cluster_n],
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'shade': True, 'linewidth': 1},
                     ax=ax2)
        ax2.tick_params(axis='x')
        ax2.set_xlim(0, 10)
        ax2.tick_params(labeltop=False)

        # creating a density plot on the X axis
        ax3 = ax[pos].twinx()
        sns.distplot(x_array[cluster_n], hist=False, kde=True, norm_hist=True,
                     bins=10, color=colors[cluster_n],
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'shade': True, 'linewidth': 1},
                     ax=ax3)

        # making the labels pretty
        ax3.tick_params(axis='y')
        ax3.tick_params(labeltop=False)
        ax3.tick_params(labelright=False)
        ax[pos].tick_params(axis='x', labelrotation=45)

        # since we need different ranges for each plot, here its specified manually
        if idx == 0:
            ax3.set_ylim(0, 1e-8)  # transit
            ax[pos].set_xlim(0, 1e10)  # transit
        if idx == 1:
            ax3.set_ylim(0, 5e-4)  # income
            ax[pos].set_xlim(0, 1e5)  # income
        elif idx == 2:
            ax[pos].set_xlim(0, 2500) # density
            ax3.set_ylim(0, 6e-2)  # density



def make_all_plots(path, plot_variable):
    pos = 1  # for aligning plots bc we need a place for R0 plot
    x_array = plot_scatter_r0(path, plot_variable)

    for date_plot in dates:
        print(date_plot)
        plot_scatter_radj(path, date_plot, pos, plot_variable, x_array)
        pos += 1


if __name__ == '__main__':
    plt.rc('font', serif='Helvetica Neue')
    plt.rcParams.update({'font.size': 16})

    # lest of dates for which plots should be generated
    dates = ['3/15/20', '3/25/20', '4/1/20', '4/10/20', '5/28/20']
    plot_variable = ['transit_scores - population weighted averages aggregated from town/city level to county',
                     'Median_Household_Income_2018',
                     'Density per square mile of land area - Housing units']
    pretty_titles = ['Relating Public Transit with Reproductive Ratio over Time',
                     'Relating Median Household Income with Reproductive Ratio over Time',
                     'Relating Density per Square Mile of Land Area with Reproductive Ratio over Time']

    path = "../results/no_validation_clusters/cluster_"
    plot_supercounties = True  # if set to False then plot on the scatter all the counties
    use_weight_average = True  # weight the supercounties over the population
    use_death_weight = False  # used for testing, futher experiments set that weighting over the deaths is not reliable
    set_transparency = 0.5  # transparency of scatter circles

    fig, ax = plt.subplots(1, len(dates) + 1, sharex=True, sharey=True)
    #for idx in range(0, len(plot_variable)):
    idx = 2
    make_all_plots(path, plot_variable[idx])
    fig.suptitle(pretty_titles[idx])
        #plt.savefig("../results/plots/"+plot_variable[idx]+'.pdf')
        #plt.savefig("../results/plots/"+plot_variable[idx]+'.png')
        #fig.tight_layout()
    plt.show()
    #    plt.draw()
    #    plt.pause(0.0001)
    #    plt.waitforbuttonpress()
    #    plt.close()
        #plt.cla()

