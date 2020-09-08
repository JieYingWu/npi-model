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
#num_of_counties = '20'

 
interventions = interventions[interventions['FIPS']%1000!=0]
regions = interventions['FIPS'].to_numpy()
save_path = 'results/table_8'

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
#print("Std matrix -----------")
#print(sd_matrix)

idx = np.argwhere(mean_matrix<3)
to_keep = [x for x in idx if x[0]!=x[1]]
print("\n\nInterventions with average less than 3-----------")
#print(to_keep)
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
