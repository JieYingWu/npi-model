import sys
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn import metrics
from os.path import join
from data_parser import get_data_state, get_data_county
import datetime as dt
from dateutil.parser import parse

base_dir = sys.argv[1]
data_dir = join(base_dir, 'data')
region = sys.argv[2]
M = int(sys.argv[3])

foot_traffic_start = dt.datetime(2020,2,15)
foot_traffic_end = dt.datetime(2020,4,11)

foot_traffic_path = join(data_dir, 'us_data', 'Google_traffic', 'retail_and_recreation_percent_change_from_baseline.csv')
foot_traffic = pd.read_csv(foot_traffic_path, index_col=0, encoding='latin1')
id_cols = ['County', 'State']
dates = [col for col in foot_traffic.columns.tolist() if col not in id_cols]
foot_traffic['scores'] = foot_traffic[dates].values.tolist()

features_path = join(data_dir, 'us_data', 'features.csv')
features = pd.read_csv(features_path, index_col=0)
features_names = [col for col in features.columns.tolist() if col not in id_cols]
features['scores'] = features[features_names].values.tolist()

if sys.argv[2] == 'US_county':
    M = int(sys.argv[3])
    stan_data, regions, start_date, geocode = get_data_county(M, data_dir, interpolate=True)

elif sys.argv[2] == 'US_state':
    M = int(sys.argv[3])
    stan_data, regions, start_date, geocode = get_data_state(M, data_dir, interpolate=True)

N2 = stan_data['N2'] 

# Estimated Rt value to predict
rt_path = join(base_dir, 'results', region+'_summary_IC.csv')
rt = pd.read_csv(rt_path, index_col=0)
rt = rt.filter(regex='^mean', axis=1)
rt = rt.filter(regex='Rt\[', axis=0)
#rt = rt.values.reshape(M, N2)# N2 values for every country, stored country 1 Rt 1:N2, country 2 Rt 1:N2...

X = None
for i in range(M):
    cur_start = parse(start_date[i])
    cur_geocode = geocode[i]
    print(cur_geocode)
    
    cur_covariates = np.stack((stan_data['covariate2'][:,i],
                                     stan_data['covariate3'][:,i], stan_data['covariate4'][:,i],
                                     stan_data['covariate5'][:,i], stan_data['covariate6'][:,i],
                                     stan_data['covariate7'][:,i], stan_data['covariate8'][:,i]), axis=1)
    cur_foot_traffic = np.array(foot_traffic.loc[cur_geocode, 'scores'])
    cur_features = np.expand_dims(features.loc[cur_geocode, 'scores'], 0)
    cur_features = np.repeat(cur_features, N2, axis=0)
        
    start_pad = (foot_traffic_start - cur_start).days
    end_pad = (cur_start + dt.timedelta(days=N2) - foot_traffic_end).days - 1
    if start_pad > 0:
        cur_foot_traffic = np.pad(cur_foot_traffic, (start_pad, end_pad), 'constant', constant_values=(0, cur_foot_traffic[-1]))
    elif start_pad < 0:
        cur_foot_traffic = cur_foot_traffic[-1*start_pad:]
        cur_foot_traffic = np.pad(cur_foot_traffic, (0, end_pad), 'constant', constant_values=(0, cur_foot_traffic[-1]))
    else:
        cur_foot_traffic = np.pad(cur_foot_traffic, (0, end_pad), 'constant', constant_values=(0, cur_foot_traffic[-1]))
    cur_foot_traffic = np.expand_dims(cur_foot_traffic, axis=1)

    cur_features[:, 0] = cur_features[:, 0] / 1000
    cur_x = np.concatenate((cur_covariates[:, 1:], cur_foot_traffic, cur_features[:, 0:]), axis=1)
    if X is None:
        X = cur_x
    else:
        X = np.concatenate((X, cur_x), axis=0)

# X columns are covariates(8), foot traffic(1), and features(2)
skip = 10
rt = rt.values[skip*N2:M*N2,:]
X = X[skip*N2:M*N2,:]
print(rt.shape, X.shape)

R_knot = 3.28

## Train the linear regression to learn alphas
y = np.log(rt)-np.log(R_knot)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#print(X_train.shape, y_train.shape)
regressor = HuberRegressor(epsilon=3, max_iter=1000)
regressor.fit(X, np.ravel(y))

print(regressor.intercept_)
print(regressor.coef_)

R_d = rt[0::N2]
dens = X[0::N2, -2]

plt.plot(dens, R_d, color='blue', linestyle='None', marker='x')
plt.xlim(0, 15)
plt.xlabel('Density')
plt.ylabel('Delta R')
plt.show()

time = np.arange(N2)
for j in range(0, M-skip, 4):
    i = j * N2
    y_hat = regressor.predict(X[i+0:i+N2, :])
    R_hat = R_knot * np.exp(y_hat)
    R_GT = rt[i+0:i+N2]

    col = np.random.rand(3,)

    plt.plot(time, R_GT, color=col, linestyle='dashed')
    plt.plot(time, R_hat, color=col, linestyle='solid')


plt.xlabel('Time (days)')
plt.ylabel('R')
plt.show()
