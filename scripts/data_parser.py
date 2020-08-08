import numpy as np
import pandas as pd
import datetime as dt

from future.backports import datetime
from os.path import join, exists
from utils import compute_moving_average
from data_preprocess import *

from enum import Enum
pd.set_option('mode.chained_assignment', None)

class Processing(Enum):
    INTERPOLATE = 0
    REMOVE_NEGATIVE_VALUES = 1
    REMOVE_NEGATIVE_REGIONS = 2


def get_cluster(data_dir, cluster):
    """ Get FIPS codes of counties in a particular cluster

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


def get_clustering(data_dir):
    """ Get cluster labels for all counties

    :param data_dir: dir where data is
    :returns: dict with key : FIPS code, value: cluster it belongs to
    :rtype: list

    """
    dtype = dict(FIPS=str, cluster=int)
    clustering = pd.read_csv(join(data_dir, 'us_data', 'clustering.csv'), dtype=dtype)
    clustering = dict(zip(clustering['FIPS'], clustering['cluster']))
    return clustering

    
def get_data(M, data_dir, processing=None, state=False, fips_list=None, validation=False,
             clustering=None, supercounties=False, validation_on_county=False, mobility=False,
             threshold=THRESHOLD, load_supercounties=False, avg_window=None, mask_term=False):
    cases, deaths, interventions, population, mobility_dict = preprocessing_us_data(data_dir)

    if state:
        cases = cases[cases['FIPS'].astype(int) % 1000 == 0]
        deaths = deaths[deaths['FIPS'].astype(int) % 1000 == 0]
    else:
        cases = cases[cases['FIPS'].astype(int) % 1000 != 0]
        deaths = deaths[deaths['FIPS'].astype(int) % 1000 != 0]

    # Not filtering interventions data since we're not selecting counties based on that
    final_dict, fips_list, dict_of_start_dates, dict_of_geo = get_regions(
        data_dir, M, cases, deaths, processing, interventions, population, mobility_dict=mobility_dict,
        fips_list=fips_list, validation=validation, supercounties=supercounties, clustering=clustering,
        mobility=mobility, threshold=threshold, load_supercounties=load_supercounties, avg_window=avg_window,
        mask_term=mask_term)

    return final_dict, fips_list, dict_of_start_dates, dict_of_geo


def save_interventions(interventions, fname):
    def func(d):
        x = d.toordinal()
        if x == 1:
            return 'NA'
        else:
            return str(x)

    interventions = interventions.copy()
    for col in interventions.columns.tolist()[3:]:
        interventions[col] = interventions[col].apply(func)
    interventions.to_csv(fname)


def load_masks(data_dir, ref=None):
    fname = join(data_dir, 'us_data', 'required_masks.csv')
    if not exists(fname):
        return None

    required_masks = pd.read_csv(fname, dtype={'FIPS': str, 'implementation': str})
    required_masks = required_masks.set_index('FIPS')

    if ref is None:
        raise NotImplementedError

    masks = ref.loc[:, ['FIPS', 'STATE', 'AREA_NAME']].set_index('FIPS')

    masks['required_masks'] = [dt.date.fromordinal(740000) for _ in range(masks.shape[0])] # * np.ones(masks.shape[0], dtype=int)
    for fips, row in required_masks.iterrows():
        for county_fips in masks.index:
            if county_fips[:2] == fips[:2]:
                masks.loc[county_fips, 'required_masks'] = dt.date(*map(int, row['implementation'].split('-')))

    return masks

    
def get_regions(data_dir, M, cases, deaths, processing, interventions, population, mobility_dict,
                fips_list=None, validation=False, clustering=None, supercounties=False,
                mobility=False, threshold=50, load_supercounties=False, avg_window=None, mask_term=False):
    if processing == Processing.INTERPOLATE:
        cases = impute(cases, allow_decrease_towards_end=False)
        deaths = impute(deaths, allow_decrease_towards_end=False)

    elif processing == Processing.REMOVE_NEGATIVE_VALUES:
        cases = remove_negative_values(cases)
        deaths = remove_negative_values(deaths)
        
    elif processing == Processing.REMOVE_NEGATIVE_REGIONS:
        cases, deaths = remove_negative_regions(cases, deaths, idx=2)

    save_tmp = fips_list is None

    print(f'mobility_dict: {mobility_dict} in get_regions')
    
    cases, deaths, interventions, population, mobility_dict = select_regions(
        cases, deaths, interventions, M, population, mobility_dict=mobility_dict, fips_list=fips_list,
        clustering=clustering, supercounties=supercounties, load_supercounties=load_supercounties)
    cases, deaths, interventions, population, mobility_dict, fips_list = select_top_regions(
        cases, deaths, interventions, M, population, mobility_dict=mobility_dict, validation=validation, threshold=threshold)

    if mask_term is None:
        masks = None
    else:
        masks = load_masks(data_dir, ref=interventions)
        masks = masks.loc[:, 'required_masks'].to_numpy()

    masks = load_masks(data_dir, ref=interventions).set_index(interventions.index)
    interventions.insert(3 + 8, 'required_masks', masks['required_masks'])

    # If mobility model, get the mobility reports
    if save_tmp:
        print('saving tmp data')
        cases.to_csv(join(data_dir, 'tmp_cases.csv'))
        deaths.to_csv(join(data_dir, 'tmp_deaths.csv'))
        save_interventions(interventions, join(data_dir, 'tmp_interventions.csv'))
    print('CASES', cases, sep='\n')
    print('DEATHS', deaths, sep='\n')
    print('INTERVENTIONS', interventions, sep='\n')
    print('POPULATION', population, sep='\n')
    print('MOBILITY', mobility_dict, sep='\n')
    
    dict_of_geo = {}            # map geocode
    for i in range(len(fips_list)):
        dict_of_geo[i] = fips_list[i]

    # drop non-numeric columns
    cases = cases.drop(['FIPS', 'Combined_Key'], axis=1)
    cases = cases.T
    cases_dates = np.array(cases.index)
    cases = cases.to_numpy()

    deaths = deaths.drop(['FIPS', 'Combined_Key'], axis=1)
    deaths = deaths.T
    deaths = deaths.to_numpy()

    if avg_window is not None:
        cases = compute_moving_average(cases, avg_window, axis=0)
        deaths = compute_moving_average(deaths, avg_window, axis=0)
        # take a avg_window-size moving average of cases and deaths.

    interventions.drop(['FIPS', 'STATE', 'AREA_NAME'], axis=1, inplace=True)
    interventions_colnames = interventions.columns.values
    covariates = interventions.to_numpy()

    population = population.drop(['FIPS'], axis=1)
    population = population.to_numpy()

    # create mobility array
    for name, df in mobility_dict.items():
        try:
            df.drop(['FIPS', 'State', 'County'], axis=1, inplace=True)
        except KeyError:
            df.drop(['FIPS'], axis=1, inplace=True)
        df.T # dates are row-wisw
        arr = df.to_numpy()
        mobility_dict[name] = arr

    mobility_report = np.dstack(tuple(mobility_dict.values()))
        
        
    if not mobility:
        mobility_report = None
    
    if validation: ### to validate model
        validation_days_dict = get_validation_dict(data_dir, cases, deaths, fips_list, cases_dates)
        deaths = apply_validation(deaths, fips_list, validation_days_dict)
        
    dict_of_start_dates, final_dict = primary_calculations(
        cases, deaths, covariates, cases_dates, population, fips_list, masks=masks, mobility=mobility_report)

    return final_dict, fips_list, dict_of_start_dates, dict_of_geo


def primary_calculations(df_cases, df_deaths, covariates, df_cases_dates, population,
                         fips_list, masks=None, mobility=None, interpolate=True):
    """"
    Returns:
        final_dict: Stan_data used to feed main sampler
        dict_of_start_dates: Starting dates considered for calculations for the top N places
    """
    
    index = np.argmax(df_cases > 0) ## make sure number of cases is greater than 0
    cum_sum = np.cumsum(df_deaths, axis=0) >= 10 ## consider only after the cumulative deaths exceed 10
    index1 = np.where(np.argmax(cum_sum, axis=0) != 0, np.argmax(cum_sum, axis=0), cum_sum.shape[0])
    index2 = index1 - 30
    start_dates = index1 + 1 - index2
    dict_of_start_dates = {}

    N2 = df_deaths.shape[0]
    
    covariate1 = []
    covariate2 = []
    covariate3 = []
    covariate4 = []
    covariate5 = []
    covariate6 = []
    covariate7 = []
    covariate8 = []
    covariate9 = []
    covariate10 = []
    covariate11 = []
    covariate12 = []
    covariate13 = []
    covariate14 = []    

    mask_covariates = []

    cases = []
    deaths = []
    N_arr = []
    mobility_list = []
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
        # N2 = 120 # initial submission
        # N2 = 180 ## decides number of days we are forecasting for
        # TODO: N2 is now set up above based on the shapes of the arrays.

        forecast = N2 - N

        if forecast < 0:
            print("FIPS: ", fips_list[i], " N: ", N)
            print("Error!!!! Number of days from data is greater than number of days for prediction!")
            raise RuntimeError('TODO: fix this scenario')
            N2 = N              # TODO: what is this doing?
        addlst = [covariates2[N - 1]] * (forecast)
        add_1 = [-1] * forecast ### padding for extra days

        if masks is not None:
            pass
            # append mask info, pad it with the last value
            # masks_req = np.where(req_dates >= dt.date.fromordinal(masks[i]), 1, 0)
            # masks_req = np.concatenate([masks_req, masks_req[-1] * np.ones(forecast)])
            # mask_covariates.append(masks_req)

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
        covariate8.append(covariates2[:, 7])  # masks required
        covariate9.append(covariates2[:, 8])  
        covariate10.append(covariates2[:, 9])  # stay at home rollback
        covariate11.append(covariates2[:, 10])  # >50 gatherings rollback
        covariate12.append(covariates2[:, 11])  # >500 gatherings rollback
        covariate13.append(covariates2[:, 12])  # restaurant dine-in rollback
        covariate14.append(covariates2[:, 13])  # entertainment/gym rollback

        # mobility
        if mobility is not None:
            # cases begin 1/22
            # mobility begins 2/15 -> difference    
            tmp = mobility[i,(i2-8):,:]
            fill_arr = np.array(3*[add_1]).T
            tmp = np.append(tmp, fill_arr, axis=0)
            mobility_list.append(tmp)

    covariate1 = np.array(covariate1).T
    covariate2 = np.array(covariate2).T
    covariate3 = np.array(covariate3).T
    covariate4 = np.array(covariate4).T
    covariate5 = np.array(covariate5).T
    covariate6 = np.array(covariate6).T
    covariate7 = np.array(covariate7).T
    covariate8 = np.array(covariate8).T
    covariate9 = np.array(covariate9).T
    covariate10 = np.array(covariate10).T
    covariate11 = np.array(covariate11).T
    covariate12 = np.array(covariate12).T
    covariate13 = np.array(covariate13).T
    covariate14 = np.array(covariate14).T    
    cases = np.array(cases).T
    deaths = np.array(deaths).T

    # the indicators
    X = np.dstack([covariate1, covariate2, covariate3, covariate4, covariate5, covariate6,
                   covariate7, covariate8, covariate9, covariate10, covariate11, covariate12,
                   covariate13, covariate14])
    X = np.moveaxis(X, 1, 0)

    if mobility is not None:
        X = np.dstack(mobility_list)
        X = np.moveaxis(X, 2, 0)
    X_partial = X

    ## populate for stan parameters
    final_dict = {}
    final_dict['M'] = len(fips_list)
    final_dict['N0'] = 6
    final_dict['P'] = 14 # num of covariates (used to be 8)
    final_dict['N'] = np.asarray(N_arr, dtype=np.int)
    final_dict['N2'] = N2
    final_dict['p'] = covariates.shape[1] - 1
    final_dict['x'] = np.arange(1, N2 + 1)
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
    final_dict['covariate9'] = covariate9
    final_dict['covariate10'] = covariate10
    final_dict['covariate11'] = covariate11
    final_dict['covariate12'] = covariate12
    final_dict['covariate13'] = covariate13
    final_dict['covariate14'] = covariate14
    final_dict['pop'] = population.astype(np.float).reshape((len(fips_list)))
    final_dict['X'] = X
    final_dict['X_partial'] = X_partial

    if masks is not None:
        mask_covariates = np.array(mask_covariates).T
        final_dict['masks'] = mask_covariates
        final_dict['masks_partial'] = mask_covariates

    ## New covariate for foot traffic data
    if mobility is not None:
        final_dict['P'] = 3
        final_dict['P_partial'] = 3
    
    return dict_of_start_dates, final_dict


if __name__ == '__main__':
    stan_data_val, regions, start_date, geocode = get_data(100, 'data', processing=Processing.REMOVE_NEGATIVE_VALUES, state=False, fips_list=None, validation=True)
    stan_data, regions, start_date, geocode = get_data(100, 'data', processing=Processing.REMOVE_NEGATIVE_VALUES, state=False, fips_list=None, validation=False)
    print(stan_data['deaths'][:,0][30:105])
    print(stan_data_val['deaths'][:,0][30:105])
    
