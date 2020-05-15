import numpy as np
import pandas as pd
import datetime as dt

from future.backports import datetime
from os.path import join, exists
from data_preprocess import *

from enum import Enum
pd.set_option('mode.chained_assignment', None)

class Processing(Enum):
    INTERPOLATE = 0
    REMOVE_NEGATIVE_VALUES = 1
    REMOVE_NEGATIVE_REGIONS = 2


def get_cluster(data_dir, cluster):
    """Get the fips codes for the 

    :param data_dir: dir where data is
    :param cluster: integer cluster label.
    :returns: list of fips codes in cluster `cluster`
    :rtype: list

    """
    dtype = dict(FIPS=str, cluster=int)
    clustering = pd.read_csv(join(data_dir, 'us_data', 'clustering.csv'), dtype=dtype)
    fips_list = list(clustering.loc[clustering['cluster'] == cluster, 'FIPS'])
    print(f'obtained {len(fips_list)} counties from cluster {cluster}')
    return fips_list

    
def get_data(M, data_dir, processing=None, state=False, fips_list=None, validation=False,
             cluster=None, supercounties=False):
    cases, deaths, interventions, population = preprocessing_us_data(data_dir)

    if state:
        cases = cases[cases['FIPS'] % 1000 == 0]
        deaths = deaths[deaths['FIPS'] % 1000 == 0]
    else:
        cases = cases[cases['FIPS'] % 1000 != 0]
        deaths = deaths[deaths['FIPS'] % 1000 != 0]

    # Not filtering interventions data since we're not selecting counties based on that
    final_dict, fips_list, dict_of_start_dates, dict_of_geo = get_regions(
        data_dir, M, cases, deaths, processing, interventions, population, fips_list,
        validation=validation, cluster=cluster, supercounties=supercounties)

    return final_dict, fips_list, dict_of_start_dates, dict_of_geo


def get_regions(data_dir, M, cases, deaths, processing, interventions, population,
                fips_list=None, validation=False, cluster=None, supercounties=False):
    if processing == Processing.INTERPOLATE:
        cases = impute(cases, allow_decrease_towards_end=False)
        deaths = impute(deaths, allow_decrease_towards_end=False)

    elif processing == Processing.REMOVE_NEGATIVE_VALUES:
        cases = remove_negative_values(cases)
        deaths = remove_negative_values(deaths)
        
    elif processing == Processing.REMOVE_NEGATIVE_REGIONS:
        cases, deaths = remove_negative_regions(cases, deaths, idx=2)
                
    if fips_list is None:
        cases, deaths, interventions, population, fips_list = select_top_regions(
            cases, deaths, interventions, M, population, supercounties=supercounties)
    else:
        cases, deaths, interventions, population = select_regions(
            cases, deaths, interventions, M, fips_list, population,
            validation=validation, cluster=cluster, supercounties=supercounties)
        cases, deaths, interventions, population, fips_list = select_top_regions(
            cases, deaths, interventions, M, population, validation=validation)

    cases.to_csv('data/tmp_cases.csv')
    deaths.to_csv('data/tmp_deaths.csv')
    print('CASES', cases, sep='\n')
    print('DEATHS', deaths, sep='\n')
    print('INTERVENTIONS', interventions, sep='\n')
    print('POPULATION', population, sep='\n')
    
    dict_of_geo = {} ## map geocode
    for i in range(len(fips_list)):
        dict_of_geo[i] = fips_list[i]

    #### drop non-numeric columns

    cases = cases.drop(['FIPS', 'Combined_Key'], axis=1)
    cases = cases.T  ### Dates are now row-wise
    cases_dates = np.array(cases.index)
    cases = cases.to_numpy()

    deaths = deaths.drop(['FIPS', 'Combined_Key'], axis=1)
    deaths = deaths.T
    deaths = deaths.to_numpy()

    interventions.drop(['FIPS', 'STATE', 'AREA_NAME'], axis=1, inplace=True)
    interventions_colnames = interventions.columns.values
    covariates = interventions.to_numpy()
    
    population = population.drop(['FIPS'], axis=1)
    population = population.to_numpy()
    
    
    if validation:
        validation_days_dict = get_validation_dict(data_dir, cases, deaths,fips_list, cases_dates)
        deaths = apply_validation(deaths, fips_list, validation_days_dict)


    dict_of_start_dates, final_dict = primary_calculations(cases, deaths, covariates, cases_dates,
            population, fips_list)

    return final_dict, fips_list, dict_of_start_dates, dict_of_geo


def primary_calculations(df_cases, df_deaths, covariates, df_cases_dates, population, fips_list, interpolate=True):
    """"
    Returns:
        final_dict: Stan_data used to feed main sampler
        dict_of_start_dates: Starting dates considered for calculations for the top N places
    """
    
    index = np.argmax(df_cases > 0)
    cum_sum = np.cumsum(df_deaths, axis=0) >= 10
    index1 = np.where(np.argmax(cum_sum, axis=0) != 0, np.argmax(cum_sum, axis=0), cum_sum.shape[0])
    index2 = index1 - 30
    start_dates = index1 + 1 - index2
    dict_of_start_dates = {}

    covariate1 = []
    covariate2 = []
    covariate3 = []
    covariate4 = []
    covariate5 = []
    covariate6 = []
    covariate7 = []
    covariate8 = []

    cases = []
    deaths = []
    N_arr = []

    for i in range(len(fips_list)):
        i2 = index2[i]
        dict_of_start_dates[i] = df_cases_dates[i2]
        case = df_cases[i2:, i]
        death = df_deaths[i2:, i]
        assert len(case) == len(death)

        req_dates = df_cases_dates[i2:]
        covariates2 = []
        req_dates = np.array([dt.datetime.strptime(x, '%m/%d/%y').date() for x in req_dates])

        ### check if interventions were in place start date onwards
        for col in range(covariates.shape[1]):
            covariates2.append(np.where(req_dates >= covariates[i, col], 1, 0))
        covariates2 = np.array(covariates2).T

        N = len(case)
        N_arr.append(N)
        N2 = 100

        forecast = N2 - N

        if forecast < 0:
            print("FIPS: ", fips_list[i], " N: ", N)
            print("Error!!!! N is greater than N2!")
            N2 = N
        addlst = [covariates2[N - 1]] * (forecast)
        add_1 = [-1] * forecast ### padding

        case = np.append(case, add_1, axis=0)
        death = np.append(death, add_1, axis=0)
        cases.append(case)
        deaths.append(death)

        covariates2 = np.append(covariates2, addlst, axis=0)
        covariate1.append(covariates2[:, 0])  # stay at home
        covariate2.append(covariates2[:, 1])  # >50 gatherings
        covariate3.append(covariates2[:, 2])  # >500 gatherings
        covariate4.append(covariates2[:, 3])  # public scools
        covariate5.append(covariates2[:, 4])  # restaurant dine-in
        covariate6.append(covariates2[:, 5])  # entertainment/gym
        covariate7.append(covariates2[:, 6])  # federal guidelines
        covariate8.append(covariates2[:, 7])  # federal guidelines
        
    covariate1 = np.array(covariate1).T
    covariate2 = np.array(covariate2).T
    covariate3 = np.array(covariate3).T
    covariate4 = np.array(covariate4).T
    covariate5 = np.array(covariate5).T
    covariate6 = np.array(covariate6).T
    covariate7 = np.array(covariate7).T
    covariate8 = np.array(covariate8).T
    cases = np.array(cases).T
    deaths = np.array(deaths).T
    
    X = np.dstack([covariate1, covariate2, covariate3, covariate4, covariate5, covariate6,
        covariate7, covariate8])
    X = np.moveaxis(X, 1, 0)
        
    final_dict = {}
    final_dict['M'] = len(fips_list)
    final_dict['N0'] = 6
    final_dict['P'] = 8 # num of covariates
    final_dict['N'] = np.asarray(N_arr, dtype=np.int)
    final_dict['N2'] = N2
    final_dict['p'] = covariates.shape[1] - 1
    final_dict['x'] = np.arange(1, N2+1)
    final_dict['cases'] = cases
    final_dict['deaths'] = deaths
    final_dict['EpidemicStart'] = np.asarray(start_dates).astype(np.int)
    final_dict['covariate1'] = covariate1
    final_dict['covariate2'] = covariate2
    final_dict['covariate3'] = covariate3
    final_dict['covariate4'] = covariate4
    final_dict['covariate5'] = covariate5
    final_dict['covariate6'] = covariate6
    final_dict['covariate7'] = covariate7
    final_dict['covariate8'] = covariate8
    final_dict['pop'] = population.astype(np.float).reshape((len(fips_list)))
    final_dict['X'] = X
    ## New covariate for foot traffic data
    
    return dict_of_start_dates, final_dict


if __name__ == '__main__':
    stan_data, regions, start_date, geocode = get_data(100, 'data', processing=Processing.REMOVE_NEGATIVE_VALUES, state=False, fips_list=None, validation=True)
    print(stan_data['deaths'])
    print(stan_data['deaths'][:,0])
    print(geocode[0])
    
