import os
from os.path import join, exists
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import json


def get_means_list(path, geo_list):
    # 1 for deaths; 0 for infections
    plot_name = "mu"
    df = pd.read_csv(path+"\\summary.csv", delimiter=',', index_col=0)
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

    print(supercounties_names)
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
            density_dict[county] = df.at[int(county), 'transit_scores - population weighted averages aggregated from town/city level to county']
            dict_r0[county] = dict_r0_supercounty[key]
    return density_dict, dict_r0


def get_rt_adj(path, geo_list):
    # 1 for deaths; 0 for infections
    plot_name = "Rt_adj["
    df = pd.read_csv(path+"\\summary.csv", delimiter=',', index_col=0)
    row_names = list(df.index.tolist())
    means = {}
    start_day = 1
    for num_of_country in range(0, len(geo_list)):
        county_number = str(int(num_of_country) + 1) + "]"
        for name in row_names:
            if plot_name in name and name.split(",")[1] == county_number:
                rowData = df.loc[name, :]
                key = geo_list[num_of_country]
                if name.split(",")[0] == (plot_name + str(start_day)):
                    means[key] = rowData['mean']
    return means


def get_start_day_dict(path, geo_list):
    dict_start_days = {}
    return dict_start_days


def plot_scatter(path):
    path_density = r"D:\JHU\corona\npi-model\npi-model\data\us_data"
    fig = plt.figure('Density')
    ax = fig.add_subplot(111)
    ax.set_xlabel("transit_scores - population weighted averages aggregated from town/city level to county")
    ax.set_ylabel("R0")
    colors = ["#377EB8", "#E41A1C", "#984EA3", "#4DAF4A", "#FF7F00"]

    for cluster_n in range(0,5):
        print(cluster_n)
        path_rt = path + str(cluster_n)
        supercounties_dist, supercounties_names = create_geocodes_dict(path_rt, path_density)
        r0_means_list = get_means_list(path_rt, supercounties_names)
        print(r0_means_list)
        #rt_adj_list = get_rt_adj(path_rt, supercounties_names)
        #print(rt_adj_list)
        #sum_density_dict = read_density(path_density, supercounties_dist)
        density_dict,dict_r0 = read_density(path_density, supercounties_dist, r0_means_list)
        print(density_dict)
        print(dict_r0)

        for key in density_dict.keys():
            x = density_dict[key]
            y = dict_r0[key]
            ax.scatter(x, y, color=colors[cluster_n], s=8, alpha=0.6)
    plt.show()


def main():

    path = r"D:\JHU\corona\npi-model\npi-model\results\table_no_validation\cluster"
    path_density = r"D:\JHU\corona\npi-model\npi-model\data\us_data"
    '''
    path_rt = path + str(0)
    supercounties_dist, supercounties_names = create_geocodes_dict(path_rt, path_density)
    r0_means_list = get_means_list(path_rt, supercounties_names)
    print(r0_means_list)
    rt_adj_list = get_rt_adj(path_rt, supercounties_names)
    start_day_dict = get_start_day_dict(path_rt, supercounties_names)
    print(start_day_dict)
    '''
    plot_scatter(path)


if __name__ == '__main__':
    # run from base directory
    main()
