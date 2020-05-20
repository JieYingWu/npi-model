import os
from os.path import join, exists
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import json
from matplotlib import gridspec
import seaborn as sns


def get_means_list(path, geo_list):
    # 1 for deaths; 0 for infections
    plot_name = "mu"
    df = pd.read_csv(path + "\\summary.csv", delimiter=',', index_col=0)
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
    df = pd.read_csv(path_rt + "\\geocode.csv", delimiter=',', index_col=0)
    supercounties_names = list(df.loc[0][:])

    with open(path_density + "\\supercounties.json") as f:
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

    #print(supercounties_names)
    return selected_dict, supercounties_names


def read_density_sum(path_density, selected_dict):
    df = pd.read_csv(path_density + "\\counties.csv", delimiter=',')
    df = df.set_index('FIPS')
    sum_density_dict = {}
    for key in selected_dict.keys():
        t_sum = 0
        for county in selected_dict[key]:
            found_number = df.at[int(county), 'Density per square mile of land area - Population']
            t_sum = t_sum + found_number
        sum_density_dict[key] = t_sum
    return sum_density_dict


def read_density(path_density, selected_dict, dict_r0_supercounty):
    df = pd.read_csv(path_density + "\\counties.csv", delimiter=',')
    df = df.set_index('FIPS')
    density_dict = {}
    dict_r0 = {}
    for key in selected_dict.keys():
        for county in selected_dict[key]:
            density_dict[county] = df.at[int(
                county), 'Density per square mile of land area - Housing units']  # / df.at[int(county),'POP_ESTIMATE_2018']
            dict_r0[county] = dict_r0_supercounty[key]
    return density_dict, dict_r0


def get_rt_adj(path, geo_list, start_day_dict):
    # 1 for deaths; 0 for infections
    plot_name = "Rt_adj["
    df = pd.read_csv(path + "\\summary.csv", delimiter=',', index_col=0)
    row_names = list(df.index.tolist())
    means = {}
    # start_day = 1
    for num_of_country in range(0, len(geo_list)):
        county_number = str(int(num_of_country) + 1) + "]"
        for name in row_names:
            if plot_name in name and name.split(",")[1] == county_number:
                rowData = df.loc[name, :]
                key = geo_list[num_of_country]
                if start_day_dict[
                    key] > 0:  # in case there is first few plots will miss some counties that haven't begun yet.
                    if name.split(",")[0] == (plot_name + str(start_day_dict[key])):
                        means[key] = rowData['mean']
                else:
                    means[key] = None
    return means


def get_start_day_dict(path, geo_list, date_plot):
    dict_start_days = {}
    df = pd.read_csv(path + "\\start_dates.csv", delimiter=',', index_col=0)
    supercounties_dates = list(df.loc[0][:])
    set_day_for_search = datetime.datetime.strptime(date_plot, '%m/%d/%y')
    for i in range(0, len(geo_list)):
        days_to_predict = (set_day_for_search - datetime.datetime.strptime(supercounties_dates[i], '%m/%d/%y')).days
        dict_start_days[geo_list[i]] = days_to_predict
    return dict_start_days


def plot_scatter_r0(path):
    pos = 0
    path_density = r"D:\JHU\corona\npi-model\npi-model\data\us_data"
    ax[pos].set_title("R0")
    colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]
    ax[pos].set_xscale('log')

    for cluster_n in range(0, 5):
        print(cluster_n)
        path_rt = path + str(cluster_n)
        supercounties_dist, supercounties_names = create_geocodes_dict(path_rt, path_density)
        r0_means_list = get_means_list(path_rt, supercounties_names)
        density_dict, dict_r0 = read_density(path_density, supercounties_dist, r0_means_list)

        y_list = []
        for key in density_dict.keys():
            x = density_dict[key]
            y = dict_r0[key]
            if y is not None:
                y_list.append(y)
            if x < 10000:
                ax[pos].scatter(x, y, color=colors[cluster_n], s=8, alpha=0.3)

        sns.distplot(y_list, hist=False, kde=True, vertical=True, norm_hist=True,
                     bins=10, color=colors[cluster_n],
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'shade': True, 'linewidth': 1},
                     ax=ax[pos])



def plot_scatter_radj(path, date_plot, pos):
    path_density = r"D:\JHU\corona\npi-model\npi-model\data\us_data"
    #ax[pos].set_ylabel("R_now")
    ax[pos].set_title(date_plot)
    #plt.xscale("log")
    colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]
    #ax2 = ax[pos].twiny()
    ax[pos].set_xscale('log')

    for cluster_n in range(0, 5):
        print(cluster_n)
        path_rt = path + str(cluster_n)
        supercounties_dist, supercounties_names = create_geocodes_dict(path_rt, path_density)

        start_day_dict = get_start_day_dict(path_rt, supercounties_names, date_plot)
        rt_adj_list = get_rt_adj(path_rt, supercounties_names, start_day_dict)

        density_dict, dict_r0 = read_density(path_density, supercounties_dist, rt_adj_list)

        y_list = []
        for key in density_dict.keys():
            x = density_dict[key]
            y = dict_r0[key]
            if y is not None:
                y_list.append(y)
            if x < 10000:
                ax[pos].scatter(x, y, color=colors[cluster_n], s=8, alpha=0.3)

        sns.distplot(y_list, hist=False, kde=True, vertical=True, norm_hist=True,
                     bins=10, color=colors[cluster_n],
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'shade': True, 'linewidth': 1},
                     ax=ax[pos])
    #ax[pos].set_xlim([0, 1000])
    #plt.xscale("log")


def main():
    path = r"D:\JHU\corona\npi-model\npi-model\results\table_no_validation\cluster"
    pos = 1  # for aligning plots
    plot_scatter_r0(path)
    for date_plot in dates:
        plot_scatter_radj(path, date_plot, pos)
        pos += 1


if __name__ == '__main__':
    dates = ['3/10/20', '3/15/20', '3/25/20', '4/1/20', '4/10/20']  # , '5/18/20']
    fig, ax = plt.subplots(1, len(dates) + 1, sharex=True, sharey=True)
    main()
    fig.suptitle('Density per square mile of land area - Housing units')
    plt.show()
    plt.savefig('grid_figure.pdf')