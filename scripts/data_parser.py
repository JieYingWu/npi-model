import os
import csv
import sys
import numpy as np
import pandas as pd
from future.backports import datetime
import datetime as dt
from os.path import join, exists
from scipy.ndimage.interpolation import shift

pd.set_option('mode.chained_assignment', None)

def get_stan_parameters_europe(data_dir, show):
    """
    Returns in a dict:
    M; // number of countries
    N0; // number of days for which to impute infections
    N[M]; // days of observed data for country m. each entry must be <= N2
    N2; // days of observed data + # of days to forecast
    x[N2]; // index of days (starting at 1)
    cases[N2,M]; // reported cases
    deaths[N2, M]; // reported deaths -- the rows with i > N contain -1 and should be ignored
    EpidemicStart[M];
    p; //intervention dates
    covariate1, ...., covariate7 //covariate variables

    """
    imp_covid_dir = join(data_dir, 'europe_data/COVID-19-up-to-date.csv')
    imp_interventions_dir = join(data_dir, 'europe_data/interventions.csv')

    interventions = pd.read_csv(imp_interventions_dir, encoding='latin-1')
    covid_up_to_date = pd.read_csv(imp_covid_dir, encoding='latin-1')

    mod_interventions = interventions.iloc[0:11, 0:8] ##only meaningful data for interventions

    mod_interventions.sort_values('Country', inplace=True)

    ### if intervention dates are after lockdown, set them to lockdown date
    for col in mod_interventions.columns.tolist():
        if col == 'Country' or col == 'lockdown':
            continue
        col1 = pd.to_datetime(mod_interventions[col], format='%Y-%m-%d').dt.date
        col2 = pd.to_datetime(mod_interventions['lockdown'], format='%Y-%m-%d').dt.date
        mod_interventions[col] = np.where(col1 > col2, col2, col1).astype(str)

    countries = mod_interventions['Country'].to_list()
    date_cols = [col for col in mod_interventions.columns.tolist() if col != 'Country']

    ###Initialize variables
    covariate1 = []
    covariate2 = []
    covariate3 = []
    covariate4 = []
    covariate5 = []
    covariate6 = []
    covariate7 = []

    cases = []
    deaths = []
    N_arr = []
    start_dates = []

    dict_of_start_dates = {}
    dict_of_geo = {}

    for country in countries:
        d1 = covid_up_to_date.loc[covid_up_to_date['countriesAndTerritories'] == country, :]
        covariates1 = mod_interventions.loc[mod_interventions['Country'] == country, date_cols]
        df_date = pd.to_datetime(d1['dateRep'], format='%d/%m/%Y').dt.date
        d1.loc[:, 'Date'] = df_date

        d1 = d1.sort_values(by=['Date']) ##sort by dates
        d1.reset_index(drop=True, inplace=True)

        ## get first day with number of cases >0
        index = (d1['cases'] > 0).idxmax()
        ## get first day with cumulative sum of deaths >=10
        index1 = (d1['deaths'].cumsum() >= 10).argmax()
        ## consider one month before
        index2 = index1 - 30
        idx = countries.index(country)
        dict_of_geo[idx] = country
        dict_of_start_dates[idx] = dt.datetime.strftime(d1['Date'][index2], format='%m-%d-%Y')
        start_dates.append(index1 + 1 - index2)

        d1 = d1.loc[index2:]
        case = d1['cases'].to_numpy()
        death = d1['deaths'].to_numpy()
        if show:
            print("{Country} has {num} days of data".format(Country=country, num=len(d1['cases'])))
            print("Start date for {Country}: ".format(Country=country), dict_of_start_dates[idx])

        ### check if interventions were in place from start date onwards
        for col in date_cols:
            covid_date = pd.to_datetime(d1['dateRep'], format='%d/%m/%Y').dt.date
            int_data = datetime.datetime.strptime(covariates1[col].to_string(index=False).strip(), '%Y-%m-%d')
            # int_date = pd.to_datetime(covariates1[col], format='%Y-%m-%d').dt.date
            d1[col] = np.where(covid_date.apply(lambda x: x >= int_data.date()), 1, 0)

        N = len(d1['cases'])
        N_arr.append(N)
        N2 = 75  ##from paper (number of days needed for prediction)
        forecast = N2 - N

        if forecast < 0:
            print("Country: ", country, " N: ", N)
            print("Error!!!! N is greater than N2!")
            N2 = N
            forecast = N2 - N

        covariates2 = d1[date_cols]
        covariates2.reset_index(drop=True, inplace=True)
        covariates2 = covariates2.to_numpy()
        addlst = [covariates2[N - 1]] * (forecast) ##padding
        add_1 = [-1] * forecast

        covariates2 = np.append(covariates2, addlst, axis=0)
        case = np.append(case, add_1, axis=0)
        death = np.append(death, add_1, axis=0)
        cases.append(case)
        deaths.append(death)

        covariate1.append(covariates2[:, 0])  # schools_universities
        covariate2.append(covariates2[:, 1])  # travel_restrictions
        covariate3.append(covariates2[:, 2])  # public_events
        covariate4.append(covariates2[:, 3])  # sports
        covariate5.append(covariates2[:, 4])  # lockdwon
        covariate6.append(covariates2[:, 5])  # social_distancing
        covariate7.append(covariates2[:, 6])  # self-isolating if ill

    covariate1 = np.array(covariate1).T
    covariate2 = np.array(covariate2).T
    covariate3 = np.array(covariate3).T
    covariate4 = np.array(covariate4).T
    covariate5 = np.array(covariate5).T
    covariate6 = np.array(covariate6).T
    covariate7 = np.array(covariate7).T
    cases = np.array(cases).T
    deaths = np.array(deaths).T

    #covariate2 = 0 * covariate2  # remove travel ban
    #covariate5 = 0 * covariate5  # remove sports
    covariate2 = covariate7  # self-isolating if ill
    covariate4 = np.where(covariate1 + covariate3 + covariate5 + covariate6 + covariate7 >= 1, 1, 0)  # any intervention

    covariate7 = 0  # models should take only one covariate

    final_dict = {}

    filename1 = 'europe_start_dates.csv'
    filename2 = 'europe_geocode.csv'
    df_sd = pd.DataFrame(dict_of_start_dates, index=[0])
    df_geo = pd.DataFrame(dict_of_geo, index=[0])
    df_sd.to_csv('results/' + filename1, sep=',')
    df_geo.to_csv('results/' + filename2, sep=',')

    final_dict['M'] = len(countries)
    final_dict['N0'] = 6
    final_dict['N'] = np.asarray(N_arr).astype(np.int)
    final_dict['N2'] = N2
    final_dict['x'] = np.arange(1, N2+1)
    final_dict['cases'] = cases
    final_dict['deaths'] = deaths
    final_dict['EpidemicStart'] = np.asarray(start_dates)
    final_dict['p'] = len(mod_interventions.columns) - 1
    final_dict['covariate1'] = covariate1
    final_dict['covariate2'] = covariate2
    final_dict['covariate3'] = covariate3
    final_dict['covariate4'] = covariate4
    final_dict['covariate5'] = covariate5
    final_dict['covariate6'] = covariate6
    final_dict['covariate7'] = covariate7

    return final_dict, countries

def get_stan_parameters_by_county_us(num_counties, data_dir, show):

    cases_path = join(data_dir, 'us_data/infections_timeseries.csv')
    deaths_path = join(data_dir, 'us_data/deaths_timeseries.csv')
    interventions_path = join(data_dir, 'us_data/interventions.csv')

    df_cases = pd.read_csv(cases_path)
    df_deaths = pd.read_csv(deaths_path)
    interventions = pd.read_csv(interventions_path)

    interventions = interventions[interventions['FIPS'] % 1000 != 0]
    id_cols = ['FIPS', 'STATE', 'AREA_NAME']
    int_cols = [col for col in interventions.columns.tolist() if col not in id_cols]
    interventions.fillna(1, inplace=True)
    for col in int_cols: ### convert date from given format
        interventions[col] = interventions[col].apply(lambda x: dt.date.fromordinal(int(x)))

    issues_cases = county_monotonicity(df_cases)
    issues_deaths = county_monotonicity(df_deaths)
    def get_daily_counts(L):
        diff = np.array([y - x for x, y in zip(L, L[1:])])
        L[1:] = diff
        return L

    #### get daily counts instead of cumulative
    df_cases.iloc[:, 2:] = df_cases.iloc[:, 2:].apply(get_daily_counts, axis=1)
    df_deaths.iloc[:, 2:] = df_deaths.iloc[:, 2:].apply(get_daily_counts, axis=1)

    # print(df_cases.columns)
    sanity_check = df_cases.iloc[:, 2:].apply(lambda x: np.sum(x < 0), axis=1)
    drop_counties = sanity_check[sanity_check != 0].index
    df_cases = df_cases.drop(drop_counties)

    sanity_check2 = df_deaths.iloc[:, 2:].apply(lambda x: np.sum(x < 0), axis=1)
    drop_counties = sanity_check2[sanity_check2 != 0].index
    df_deaths = df_deaths.drop(drop_counties)

    ## filter only the FIPS that are present in both cases and deaths timeseries
    intersect = list(set(df_cases['FIPS']) & set(df_deaths['FIPS']))
    df_cases = df_cases[df_cases['FIPS'].isin(intersect)]
    df_deaths = df_deaths[df_deaths['FIPS'].isin(intersect)]

    # Pick top 20 counties with most cases
    headers = df_cases.columns.values
    last_day = headers[-1]
    observed_days = len(headers[2:])

    df_cases = df_cases.sort_values(by=[last_day], ascending=False)
    df_cases = df_cases.iloc[:num_counties].copy()
    df_cases = df_cases.reset_index(drop=True)

    dict_of_geo = {}
    fips_list = df_cases['FIPS'].tolist()
    for i in range(len(fips_list)):
        #comb_key = df_cases.loc[df_cases['FIPS'] == fips_list[i], 'Combined_Key'].to_string(index=False)
        dict_of_geo[i] = fips_list[i]

    merge_df = pd.DataFrame({'merge': fips_list})
    df_deaths = df_deaths.loc[df_deaths['FIPS'].isin(fips_list)]
    # Select the 20 counties in the same order from the deaths dataframe by merging
    df_deaths = pd.merge(merge_df, df_deaths, left_on='merge', right_on='FIPS', how='outer')
    df_deaths = df_deaths.reset_index(drop=True)

    interventions = interventions.loc[interventions['FIPS'].isin(fips_list)]
    interventions = pd.merge(merge_df, interventions, left_on='merge', right_on='FIPS', how='outer')
    interventions = interventions.reset_index(drop=True)
    interventions = interventions.drop(['merge'], axis=1)

    df_cases = df_cases.drop(['FIPS', 'Combined_Key'], axis=1)
    df_cases = df_cases.T  ### Dates are now row-wise
    df_cases_dates = np.array(df_cases.index)
    df_cases = df_cases.to_numpy()

    df_deaths = df_deaths.drop(['merge', 'FIPS', 'Combined_Key'], axis=1)
    df_deaths = df_deaths.T
    df_deaths = df_deaths.to_numpy()

    interventions.drop(id_cols, axis=1, inplace=True)
    interventions_colnames = interventions.columns.values
    covariates1 = interventions.to_numpy()

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
        for col in range(covariates1.shape[1]):
            covariates2.append(np.where(req_dates >= covariates1[i, col], 1, 0))
        covariates2 = np.array(covariates2).T

        if show:
            print("County with FIPS {fips} has {num} days of data".format(fips=fips_list[i], num=len(case)))
            print("Start date: ", dict_of_start_dates[i])

        N = len(case)
        N_arr.append(N)
        N2 = 100

        forecast = N2 - N

        if forecast < 0:
            print("FIPS: ", fips_list[i], " N: ", N)
            print("Error!!!! N is greater than N2!")
            N2 = N
        addlst = [covariates2[N - 1]] * (forecast)
        add_1 = [-1] * forecast

        case[case < 0] = 0
        case = np.append(case, add_1, axis=0)
        death[death < 0] = 0
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

    covariate1 = np.array(covariate1).T
    covariate2 = np.array(covariate2).T
    covariate3 = np.array(covariate3).T
    covariate4 = np.array(covariate4).T
    covariate5 = np.array(covariate5).T
    covariate6 = np.array(covariate6).T
    covariate7 = np.array(covariate7).T
    cases = np.array(cases).T
    deaths = np.array(deaths).T
    #print(np.sum(cases<-1))
    #print(np.sum(deaths<-1))

    filename1 = 'us_county_start_dates.csv'
    filename2 = 'us_county_geocode.csv'
    df_sd = pd.DataFrame(dict_of_start_dates, index=[0])
    df_geo = pd.DataFrame(dict_of_geo, index=[0])
    df_sd.to_csv('results/' + filename1, sep=',')
    df_geo.to_csv('results/' + filename2, sep=',')

    final_dict = {}
    final_dict['M'] = num_counties
    final_dict['N0'] = 6
    final_dict['N'] = np.asarray(N_arr, dtype=np.int)
    final_dict['N2'] = N2
    final_dict['x'] = np.arange(1, N2+1)
    final_dict['cases'] = cases
    final_dict['deaths'] = deaths
    final_dict['EpidemicStart'] = np.asarray(start_dates).astype(np.int)
    final_dict['p'] = len(interventions_colnames) - 1
    final_dict['covariate1'] = covariate1
    final_dict['covariate2'] = covariate2
    final_dict['covariate3'] = covariate3
    final_dict['covariate4'] = covariate4
    final_dict['covariate5'] = covariate5
    final_dict['covariate6'] = covariate6
    final_dict['covariate7'] = covariate7

    return final_dict, fips_list

def check_monotonicity(L):
    is_monotonic = np.sum([x<= y for x, y in zip(L, L[1:])])
    return is_monotonic

def county_monotonicity(df):
    df1 = df.iloc[:, 2:].apply(check_monotonicity, axis=1)
    df1 = df1[df1!=(df.shape[1] - 3)]
    idx_row = {}
    return df.iloc[df1.index]['FIPS']

def get_stan_parameters_by_state_us(num_states, data_dir, show):

    cases_path = join(data_dir, 'us_data/infections_timeseries.csv')
    deaths_path = join(data_dir, 'us_data/deaths_timeseries.csv')
    interventions_path = join(data_dir, 'us_data/interventions.csv')

    df_cases = pd.read_csv(cases_path, encoding='latin1')
    df_deaths = pd.read_csv(deaths_path)

    interventions = pd.read_csv(interventions_path, encoding='latin1')
    interventions.fillna(1, inplace=True)
    interventions.drop([0], axis=0, inplace=True)
    beginning_ids_int = np.unique(np.array(interventions['FIPS'] / 1000).astype(np.int))
    id_cols = ['FIPS', 'STATE', 'AREA_NAME', 'Combined_Key']
    int_cols = [col for col in interventions.columns.tolist() if col not in id_cols]
    int_cases_col = [col for col in df_cases.columns.tolist() if col not in id_cols]
    for col in int_cols:
        interventions[col] = interventions[col].apply(lambda x: dt.date.fromordinal(int(x)))

    #### check monotonicity of the data
    #issues_cases = county_monotonicity(df_cases)
    #issues_deaths = county_monotonicity(df_deaths)

    def get_daily_counts(L):
        diff = np.array([y - x for x, y in zip(L, L[1:])])
        L[1:] = diff
        return L

    df_cases.iloc[:, 2:] = df_cases.iloc[:, 2:].apply(get_daily_counts, axis=1)
    df_deaths.iloc[:, 2:] = df_deaths.iloc[:, 2:].apply(get_daily_counts, axis=1)
    #print(df_cases.shape, df_deaths.shape)

    state_interventions = pd.DataFrame(columns=int_cols, index=beginning_ids_int * 1000)
    for i in beginning_ids_int:
        county_int = interventions.loc[(interventions['FIPS'] / 1000).astype(int) == i, :]
        ## set the latest date for intervention at any county as the date of intervention for the state
        state_interventions.loc[i * 1000, :] = county_int[int_cols].max(axis=0)

    state_interventions.insert(0, 'FIPS', state_interventions.index)

    beginning_ids_cases = np.unique(np.array(df_cases['FIPS'] / 1000).astype(np.int))
    cases_dates_cols = [col for col in df_cases.columns.tolist() if col not in id_cols]
    deaths_dates_cols = [col for col in df_deaths.columns.tolist() if col not in id_cols]
    state_cases = pd.DataFrame(columns=cases_dates_cols, index=beginning_ids_cases * 1000)
    state_deaths = pd.DataFrame(columns=deaths_dates_cols, index=beginning_ids_cases * 1000)

    ### get daily counts over states
    for i in beginning_ids_cases:
        county_case_int = df_cases.loc[(df_cases['FIPS'] / 1000).astype(int) == i, cases_dates_cols]
        county_death_int = df_deaths.loc[(df_deaths['FIPS'] / 1000).astype(int) == i, deaths_dates_cols]
        state_cases.loc[i * 1000, :] = county_case_int.sum(axis=0)
        state_deaths.loc[i * 1000, :] = county_death_int.sum(axis=0)

    state_cases.insert(0, 'FIPS', state_cases.index)
    state_deaths.insert(0, 'FIPS', state_deaths.index)

    sanity_check = state_cases.iloc[:, 1:].apply(lambda x: np.sum(x < 0), axis=1)
    drop_states = sanity_check[sanity_check != 0].index
    state_cases = state_cases.drop(drop_states)

    sanity_check2 = state_deaths.iloc[:, 1:].apply(lambda x: np.sum(x < 0), axis=1)
    drop_states = sanity_check2[sanity_check2 != 0].index
    state_deaths = state_deaths.drop(drop_states)

    ## filter only the FIPS that are present in both cases and deaths timeseries
    intersect = list(set(state_cases['FIPS']) & set(state_deaths['FIPS']))
    state_cases = state_cases[state_cases['FIPS'].isin(intersect)]
    state_deaths = state_deaths[state_deaths['FIPS'].isin(intersect)]

    # Pick top N states with most cases
    headers = state_cases.columns.values
    last_day = headers[-1]
    state_cases = state_cases.sort_values(by=[last_day], ascending=False)
    state_cases = state_cases.iloc[:num_states]
    state_cases = state_cases.reset_index(drop=True)

    dict_of_geo = {}
    fips_list = state_cases['FIPS'].tolist()

    for i in range(len(fips_list)):
        # comb_key = df_cases.loc[df_cases['FIPS'] == fips_list[i], 'Combined_Key'].to_string(index=False)
        dict_of_geo[i] = fips_list[i]

    merge_df = pd.DataFrame({'merge': fips_list})
    state_deaths = state_deaths.loc[state_deaths['FIPS'].isin(fips_list)]
    state_deaths = pd.merge(merge_df, state_deaths, left_on='merge', right_on='FIPS', how='inner')
    state_deaths = state_deaths.reset_index(drop=True)

    state_interventions = state_interventions.loc[state_interventions['FIPS'].isin(fips_list)]
    state_interventions = pd.merge(merge_df, state_interventions, left_on='merge', right_on='FIPS', how='outer')
    state_interventions = state_interventions.reset_index(drop=True)

    state_cases = state_cases.drop(['FIPS'], axis=1)
    state_cases = state_cases.T  ### Dates are now row-wise
    state_cases_dates = np.array(state_cases.index)
    state_cases = state_cases.to_numpy()

    state_deaths = state_deaths.drop(['merge', 'FIPS'], axis=1)
    state_deaths = state_deaths.T
    state_deaths = state_deaths.to_numpy()

    state_interventions.drop(['merge', 'FIPS'], axis=1, inplace=True)
    state_interventions_colnames = state_interventions.columns.values

    covariates1 = state_interventions.to_numpy()

    #### cases and deaths are already cumulative
    index = np.argmax(state_cases > 0)
    cum_sum = np.cumsum(state_deaths, axis=0) >= 10
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

    cases = []
    deaths = []
    N_arr = []

    for i in range(len(fips_list)):
        i2 = index2[i]
        dict_of_start_dates[i] = state_cases_dates[i2]
        case = state_cases[i2:, i]
        death = state_deaths[i2:, i]
        req_dates = state_cases_dates[i2:]
        covariates2 = []
        req_dates = np.array([dt.datetime.strptime(x, '%m/%d/%y').date() for x in req_dates])

        ### check if interventions were in place start date onwards
        for col in range(covariates1.shape[1]):
            covariates2.append(np.where(req_dates >= covariates1[i, col], 1, 0))
        covariates2 = np.array(covariates2).T

        if show:
            print("State with FIPS {fips} has {num} days of data".format(fips=fips_list[i], num=len(case)))
            print("Start date: ", dict_of_start_dates[i])

        N = len(case)
        N_arr.append(N)
        N2 = 100

        forecast = N2 - N

        if forecast < 0:
            print("FIPS: ", fips_list[i], " N: ", N)
            print("Error!!!! N is greater than N2!")
            N2 = N
        addlst = [covariates2[N - 1]] * (forecast)
        add_1 = [-1] * forecast

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

    covariate1 = np.array(covariate1).T
    covariate2 = np.array(covariate2).T
    covariate3 = np.array(covariate3).T
    covariate4 = np.array(covariate4).T
    covariate5 = np.array(covariate5).T
    covariate6 = np.array(covariate6).T
    covariate7 = np.array(covariate7).T
    cases = np.array(cases).T.astype(int)
    deaths = np.array(deaths).T.astype(int)
    #print(np.sum(cases < -1))
    #print(np.sum(deaths < -1))

    filename1 = 'us_states_start_dates.csv'
    filename2 = 'us_states_geocode.csv'
    df_sd = pd.DataFrame(dict_of_start_dates, index=[0])
    df_geo = pd.DataFrame(dict_of_geo, index=[0])
    df_sd.to_csv('results/' + filename1, sep=',')
    df_geo.to_csv('results/' + filename2, sep=',')

    final_dict = {}
    final_dict['M'] = num_states
    final_dict['N0'] = 6
    final_dict['N'] = np.asarray(N_arr, dtype=np.int)
    final_dict['N2'] = N2
    final_dict['x'] = np.arange(1, N2+1)
    final_dict['cases'] = cases
    final_dict['deaths'] = deaths
    final_dict['EpidemicStart'] = np.asarray(start_dates).astype(np.int)
    final_dict['p'] = len(state_interventions_colnames)-1
    final_dict['covariate1'] = covariate1
    final_dict['covariate2'] = covariate2
    final_dict['covariate3'] = covariate3
    final_dict['covariate4'] = covariate4
    final_dict['covariate5'] = covariate5
    final_dict['covariate6'] = covariate6
    final_dict['covariate7'] = covariate7

    return final_dict, fips_list

# if __name__ == '__main__':
#
#     data_dir = 'data'
#     ## Europe data
#     get_stan_parameters_europe(data_dir, show=True)
#     print("***********************")
#     ## US data
#     get_stan_parameters_by_state_us(5, data_dir, show=True)
#     print("***********************")
#     get_stan_parameters_by_county_us(5, data_dir, show=True)

