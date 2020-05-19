# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:25:08 2020

@author: shreya
"""
import os
import pandas as pd
import numpy as np

fips_list = [1073, 8115, 13095, 15007, 18051, 29095, 37049, 48041, 48157, 48439,
               55079, 53033, 42101, 47089, 36119, 50027, 20101, 29077, 34025, 18097]

ts_cases_fake = pd.read_csv('simulated/us_data/infections_timeseries_w_states.csv')
ts_deaths_fake = pd.read_csv('simulated/us_data/deaths_timeseries_w_states.csv')
check_cases = ts_cases_fake.loc[ts_cases_fake['FIPS'].isin(fips_list)]
check_cases = check_cases.to_numpy()
check_deaths = ts_deaths_fake.loc[ts_deaths_fake['FIPS'].isin(fips_list)]
check_deaths = check_deaths.to_numpy()

n_cases = check_cases.shape[1]
n_deaths = check_deaths.shape[1]

real_cases = pd.read_csv('results/real_county/summary.csv', index_col = 0)
results_mean = real_cases['mean']
params = results_mean.index
cases = [results_mean[x] for x in params if x.startswith('prediction[')]
deaths = [results_mean[x] for x in params if x.startswith('E_deaths[')]
rt = [results_mean[x] for x in params if x.startswith('Rt[')]

cases = np.array(cases).reshape(150, 20).T
deaths = np.array(deaths).reshape(150, 20).T
rt = np.array(rt)

cases = cases[:, :n_cases]
deaths = deaths[:, :n_deaths]

print(cases)
#print(np.sum(check_cases==cases))
#print(np.sum(check_cases==deaths))
