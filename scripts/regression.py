import sys
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from os.path import join

data_dir = sys.argv[1]
foot_traffic = np.genfromtxt(join(data_dir, 'us_data', 'retail_recreation_fips.csv', delimiter=',', skip_header=1))

interventions_path = join(data_dir, 'us_data', 'interventions.csv')
interventions = pd.read_csv(interventions_path)

id_cols = ['FIPS', 'STATE', 'AREA_NAME']
int_cols = [col for col in interventions.columns.tolist() if col not in id_cols]

interventions.drop([0], axis=0, inplace=True)
interventions.fillna(1, inplace=True)

for col in int_cols: ### convert date from given format
    interventions[col] = interventions[col].apply(lambda x: dt.date.fromordinal(int(x)))
