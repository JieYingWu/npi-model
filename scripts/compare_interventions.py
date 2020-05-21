# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:50:54 2020

@author: shreya
"""

import numpy as np
import pandas as pd
from os.path import join
import datetime as dt
#from data_parser import get_data, Processing
from datetime import datetime
import sys


interventions_path = join('data', 'us_data/interventions.csv')
interventions = pd.read_csv(interventions_path)

#num_of_counties = sys.argv[1]
num_of_counties = '20'

if num_of_counties == '20':
    regions = [1073, 8115, 13095, 15007, 18051, 29095, 37049, 48041, 48157, 48439,
                   55079, 53033, 42101, 47089, 36119, 50027, 20101, 29077, 34025, 18097]
    save_path = 'results/compare_interventions/20_counties'
    
elif num_of_counties == 'all':
    interventions = interventions[interventions['FIPS']%1000!=0]
    regions = interventions['FIPS'].to_numpy()
    save_path = 'results/compare_interventions/all_counties'

regions.sort()
# print(interventions.columns)
interventions = interventions[interventions['FIPS'].isin(regions)]
id_cols = ['FIPS', 'STATE', 'AREA_NAME']    
int_cols = [col for col in interventions.columns.tolist() if col not in id_cols]

for col in int_cols: ### convert date from given format
    interventions[col] = interventions[col].apply(lambda x: dt.date.fromordinal(int(x)))

interventions.drop(id_cols, axis = 1, inplace=True)
_, n = interventions.shape
mean_matrix = np.zeros((n, n))
sd_matrix = np.zeros((n, n))

print(interventions.head())
for i in range(n):
    for j in range(n):
        find_diff = abs(interventions.iloc[:, i] - interventions.iloc[:, j]).dt.days
        diff_int = find_diff[find_diff<100]
        diff_int = diff_int.to_numpy()
        avg = np.average(diff_int)
        sd = np.std(diff_int)
        mean_matrix[i, j] = avg
        sd_matrix[i, j] = sd
        
print("Mean matrix ----------")
print(mean_matrix)
print("Std matrix -----------")
print(sd_matrix)

idx = np.argwhere(mean_matrix<3)
to_keep = [x for x in idx if x[0]!=x[1]]
print("\n\nInterventions with average less than 3-----------")
print(to_keep)
df_mean = pd.DataFrame(mean_matrix)
df_mean.columns = int_cols
df_mean.index = int_cols

#print(df_mean)
df_mean.to_csv(join(save_path, 'interventions_mean.csv'))

df_std = pd.DataFrame(sd_matrix)
df_std.columns = int_cols
df_std.index = int_cols

#print(df_std)
df_std.to_csv(join(save_path, 'interventions_std.csv'))

# for key, val in diff.items():
#     if val<4:
#         print(key)
# print(diff)
# check_all_vals = list(diff.values())
# check_all_vals = np.array(check_all_vals)
# ultimates = check_all_vals[check_all_vals]

# stan_data, regions, start_date, geocode = get_data(len(regions), data_dir='data', processing=Processing.REMOVE_NEGATIVE_VALUES, state=False, fips_list=regions)
# i1 = stan_data['covariate1']
# i2 = stan_data['covariate2']
# i3 = stan_data['covariate3']
# i4 = stan_data['covariate4']
# i5 = stan_data['covariate5']
# i6 = stan_data['covariate6']
# i7 = stan_data['covariate7']
# i8 = stan_data['covariate8']

# # interventions = np.concatenate((i1, i2, i3, i4, i5, i6, i7, i8), axis=2)

# convert_to_dates = [datetime.strptime(x, '%m/%d/%y') for x in list(start_date.values())]
# latest_date = max(convert_to_dates)
# convert_to_dates = np.array(convert_to_dates)
# check_diff = abs(convert_to_dates - latest_date)
# check_diff = [x.days for x in check_diff]

# stay_at_home = i1[check_diff[1]:, :]
# more_than_50 = i2[check_diff[2]:, :]
# more_than_500 = i3[check_diff[3]:, :]
# public_school = i4[check_diff[4]:, :]
# dine_in = i4[check_diff[5]:, :]
# entertainment = i5[check_diff[6]:, :]
# fed_guidelines = i6[check_diff[7]:, :]
# federal_guidelines = i7[check_diff[8]:, :]

# print(stay_at_home.shape, more_than_50.shape)
    
# num_of_sim = {}
# for key, value in diff.items():
#     num_of_sim[key] = len(value[value<=2])
# print(num_of_sim)


