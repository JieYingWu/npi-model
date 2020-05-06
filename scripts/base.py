from os.path import join
import sys
import numpy as np
from data_parser import get_data, Processing
from data_parser_europe import get_data_europe
import pystan
import datetime as dt
from dateutil.parser import parse
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

assert len(sys.argv) < 6

# Compile the model
data_dir = sys.argv[1]
if sys.argv[2] == 'europe':
    stan_data, regions, start_date, geocode = get_data_europe(data_dir, show=False)
    weighted_fatalities = np.loadtxt(join(data_dir, 'europe_data', 'weighted_fatality.csv'), skiprows=1, delimiter=',', dtype=str)

elif sys.argv[2] == 'US_county':
    M = int(sys.argv[3])
    stan_data, regions, start_date, geocode = get_data(M, data_dir, processing=Processing.REMOVE_NEGATIVE_VALUES, state=False)
    wf_file = join(data_dir, 'us_data', 'weighted_fatality.csv')
    weighted_fatalities = np.loadtxt(wf_file, skiprows=1, delimiter=',', dtype=str)

elif sys.argv[2] == 'US_state':
    M = int(sys.argv[3])
    stan_data, regions, start_date, geocode = get_data(M, data_dir, processing=Processing.REMOVE_NEGATIVE_VALUES, state=True)
    wf_file = join(data_dir, 'us_data', 'state_weighted_fatality.csv')
    weighted_fatalities = np.loadtxt(wf_file, skiprows=1, delimiter=',', dtype=str)

N2 = stan_data['N2']

# Build a dictionary of region identifier to weighted fatality rate
ifrs = {}
for i in range(weighted_fatalities.shape[0]):
    ifrs[weighted_fatalities[i,0]] = weighted_fatalities[i,-1]
stan_data['cases'] = stan_data['cases'].astype(np.int)
stan_data['deaths'] = stan_data['deaths'].astype(np.int)
# np.savetxt('cases.csv', stan_data['cases'].astype(int), delimiter=',', fmt='%i')
# np.savetxt('deaths.csv', stan_data['deaths'].astype(int), delimiter=',', fmt='%i')

# Build a dictionary for shelter-in-place score for US cases, also load correct model for region
if sys.argv[2][0:2] == 'US':
#     foot_traffic_path = join(data_dir, 'us_data', 'Google_traffic', 'retail_and_recreation_percent_change_from_baseline.csv')
#     foot_traffic = pd.read_csv(foot_traffic_path, index_col=0, encoding='latin1')
#     id_cols = ['County', 'State']
#     dates = [col for col in foot_traffic.columns.tolist() if col not in id_cols]
#     foot_traffic['scores'] = foot_traffic[dates].values.tolist()

#     foot_traffic_start = dt.datetime(2020, 2, 15)
#     foot_traffic_end = dt.datetime(2020, 4, 11)
#     covariate9 = np.zeros((N2, M))

#     features_path = join(data_dir, 'us_data', 'features.csv')
#     features = pd.read_csv(features_path, index_col=0)
#     covariate10 = np.zeros((N2, M))
#     covariate11 = np.zeros((N2, M))
    
#     for i in range(len(regions)):
#         r = regions[i]
#         cur_start = parse(start_date[i])
#         cur_foot_traffic = np.array(foot_traffic.loc[r, 'scores'])

#         start_pad = (foot_traffic_start - cur_start).days
#         end_pad = (cur_start + dt.timedelta(days=N2) - foot_traffic_end).days - 1
#         if start_pad > 0:
#             cur_foot_traffic = np.pad(cur_foot_traffic, (start_pad, end_pad), 'constant', constant_values=(0, cur_foot_traffic[-1]))
#         elif start_pad < 0:
#             cur_foot_traffic = cur_foot_traffic[-1*start_pad:]
#             cur_foot_traffic = np.pad(cur_foot_traffic, (0, end_pad), 'constant', constant_values=(0, cur_foot_traffic[-1]))
#         else:
#             cur_foot_traffic = np.pad(cur_foot_traffic, (0, end_pad), 'constant', constant_values=(0, cur_foot_traffic[-1]))

#         density = features.loc[r, 'Density per square mile of land area - Population']
#         code = features.loc[r, 'Rural-urban_Continuum Code_2013']
        
#         covariate9[:, i] = cur_foot_traffic
#         covariate10[:, i] = np.repeat([density], N2)
# #        covariate11[:, i] = np.repeat([code], N2)
#     stan_data['covariate9'] = covariate9
#     stan_data['covariate10'] = covariate10
#    stan_data['covariate11'] = covariate11
    # Train the model and generate samples - returns a StanFit4Model
    sm = pystan.StanModel(file='stan-models/us_new.stan')
else:
    # Train the model and generate samples - returns a StanFit4Model
    sm = pystan.StanModel(file='stan-models/base_europe.stan')

serial_interval = np.loadtxt(join(data_dir, 'serial_interval.csv'), skiprows=1, delimiter=',')
# Time between primary infector showing symptoms and secondary infected showing symptoms - this is a probability distribution from 1 to 100 days

SI = serial_interval[0:stan_data['N2'],1]
stan_data['SI'] = SI

# infection to onset
mean1 = 5.1
cv1 = 0.86
alpha1 = cv1**-2
beta1 = mean1/alpha1
# onset to death
mean2 = 18.8
cv2 = 0.45
alpha2 = cv2**-2
beta2 = mean2/alpha2

all_f = np.zeros((N2, len(regions)))
for r in range(len(regions)):
    ifr = float(ifrs[str(regions[r])])
    
    ## assume that IFR is probability of dying given infection
    x1 = np.random.gamma(alpha1, beta1, 5000000) # infection-to-onset -> do all people who are infected get to onset?
    x2 = np.random.gamma(alpha2, beta2, 5000000) # onset-to-death
    f = ECDF(x1+x2)
    def conv(u): # IFR is the country's probability of death
        return ifr * f(u)

    h = np.zeros(N2) # Discrete hazard rate from time t = 1, ..., 100
    h[0] = (conv(1.5) - conv(0.0))

    for i in range(1, N2):
        h[i] = (conv(i+.5) - conv(i-.5)) / (1-conv(i-.5))
    s = np.zeros(N2)
    s[0] = 1
    for i in range(1, N2):
        s[i] = s[i-1]*(1-h[i-1])

    all_f[:,r] = s * h

stan_data['f'] = all_f

    
fit = sm.sampling(data=stan_data, iter=1000, chains=4, warmup=500, thin=4, control={'adapt_delta':0.9, 'max_treedepth':12})
# fit = sm.sampling(data=stan_data, iter=2000, chains=4, warmup=10, thin=4, seed=101, control={'adapt_delta':0.9, 'max_treedepth':10})

print("Finished fitting")

summary_dict = fit.summary()
print("Summarized model")

df = pd.DataFrame(summary_dict['summary'],
                 columns=summary_dict['summary_colnames'],
                 index=summary_dict['summary_rownames'])


df.to_csv('results/' + sys.argv[2] + '_summary_411.csv', sep=',')

df_sd = pd.DataFrame(start_date, index=[0])
df_geo = pd.DataFrame(geocode, index=[0])
df_sd.to_csv('results/' + sys.argv[2] + '_start_dates_411.csv', sep=',')
df_geo.to_csv('results/' + sys.argv[2] + '_geocode_411.csv', sep=',')
