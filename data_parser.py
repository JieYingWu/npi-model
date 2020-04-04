import os
import csv
import sys
import numpy as np
import pandas as pd
from future.backports import datetime
import datetime as dt

def get_stan_parameters(save_new_csv=True):
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
    covariate1, ...., covariate7

    """
    imp_covid_dir = 'data/COVID-19-up-to-date.csv'
    imp_interventions_dir = 'data/interventions.csv'
    
    interventions = pd.read_csv(imp_interventions_dir, encoding='latin-1')
    covid_up_to_date = pd.read_csv(imp_covid_dir, encoding='latin-1')

    mod_interventions = pd.DataFrame(columns=['Country', 'school/uni closures', 'self-isolating if ill',
                                              'banning public events', 'any government intervention',
                                              'complete/partial lockdown', 'social distancing/isolation'])

    mod_interventions['Country'] = interventions.iloc[0:11]['Country']
    mod_interventions['school/uni closures'] = interventions.iloc[0:11]['schools_universities']
    mod_interventions['self-isolating if ill'] = interventions.iloc[0:11]['self_isolating_if_ill']
    mod_interventions['banning public events'] = interventions.iloc[0:11]['public_events']
    mod_interventions['social distancing/isolation'] = interventions.iloc[0:11]['social_distancing_encouraged']
    mod_interventions['complete/partial lockdown'] = interventions.iloc[0:11]['lockdown']
    mod_interventions['any government intervention'] = interventions.iloc[0:11]['lockdown']
    mod_interventions['sport'] = interventions.iloc[0:11]['sport']
    mod_interventions['travel_restrictions'] = interventions.iloc[0:11]['travel_restrictions']

    mod_interventions.sort_values('Country', inplace=True)

    for col in mod_interventions.columns.tolist():
        if col == 'Country' or col == 'complete/partial lockdown':
            continue
        col1 = pd.to_datetime(mod_interventions[col], format='%Y-%m-%d').dt.date
        col2 = pd.to_datetime(mod_interventions['complete/partial lockdown'], format='%Y-%m-%d').dt.date
        col3 = pd.to_datetime(mod_interventions['any government intervention'], format='%Y-%m-%d').dt.date
        mod_interventions[col] = np.where(col1 > col2, col2, col1).astype(str)
        if col != 'self-isolating if ill':
            mod_interventions['any government intervention'] = np.where(col1 < col3, col1, col1).astype(str)

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


    for country in countries:
        d1 = covid_up_to_date.loc[covid_up_to_date['countriesAndTerritories'] == country]
        covariates1 = mod_interventions.loc[mod_interventions['Country'] == country][date_cols]
        d1['Date'] = pd.to_datetime(d1['dateRep'], format='%d/%m/%Y').dt.date
        ## No idea why they needed another date column

        d1.sort_values(by=['Date'], inplace=True)
        d1.reset_index(drop=True, inplace=True)

        ## get first day with number of cases >0
        index = (d1['cases'] > 0).idxmax()
        index1 = (d1['deaths'].cumsum() >= 10).argmax()
        index2 = index1 - 30

        start_dates.append(index1 + 1 - index2)

        d1 = d1.loc[index2:]
        case = d1['cases'].to_numpy()
        death = d1['deaths'].to_numpy()
        print("{Country} has {num} days of data".format(Country=country, num=len(d1['cases'])))

        for col in date_cols:
            covid_date = pd.to_datetime(d1['dateRep'], format='%d/%m/%Y').dt.date
            int_data = datetime.datetime.strptime(covariates1[col].to_string(index=False).strip(), '%Y-%m-%d')
            # int_date = pd.to_datetime(covariates1[col], format='%Y-%m-%d').dt.date
            d1[col] = np.where(covid_date.apply(lambda x: x >= int_data.date()), 1, 0)

        N = len(d1['cases'])
        N_arr.append(N)
        N2 = 65  ##from paper
        forecast = N2 - N

        if forecast < 0:
            print("Country: ", country, " N: ", N)
            print("Error!!!! N2 is increasing!")
            N2 = N
            forecast = N2 - N

        covariates2 = d1[date_cols]
        covariates2.reset_index(drop=True, inplace=True)
        covariates2 = covariates2.to_numpy()
        addlst = [covariates2[N - 1]] * (forecast)
        add_1 = [-1] * forecast

        covariates2 = np.append(covariates2, addlst, axis=0)
        case = np.append(case, add_1, axis=0)
        death = np.append(death, add_1, axis=0)
        cases.append(case)
        deaths.append(death)

        covariate1.append(covariates2[:, 0])  # school_universities
        covariate2.append(covariates2[:, 7])  # travel_restrictions
        covariate3.append(covariates2[:, 2])  # public_events
        covariate4.append(covariates2[:, 6])  # sports
        covariate5.append(covariates2[:, 4])  # lockdwon
        covariate6.append(covariates2[:, 5])  # social_distancing
        covariate7.append(covariates2[:, 1])  # self-isolating if ill

        # converting to numpy array
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
    final_dict['M'] = len(countries)
    final_dict['N0'] = 6
    final_dict['N'] = np.asarray(N_arr).astype(np.int)
    final_dict['N2'] = N2
    final_dict['x'] = np.arange(0, N2)
    final_dict['cases'] = cases
    final_dict['deaths'] = deaths
    final_dict['EpidemicStart'] = np.asarray(start_dates)
    final_dict['p'] = len(mod_interventions.columns) - 2  # s.t government intervention is not included
    final_dict['covariate1'] = covariate1
    final_dict['covariate2'] = covariate2
    final_dict['covariate3'] = covariate3
    final_dict['covariate4'] = covariate4
    final_dict['covariate5'] = covariate5
    final_dict['covariate6'] = covariate6
    final_dict['covariate7'] = covariate7
    return final_dict, countries



def get_stan_parameters_our(num_counties):
    
    #### Set directory for our data
    cases_path = '../disease_spread/data/infections_timeseries.csv'
    deaths_path = '../disease_spread/data/deaths_timeseries.csv'
    interventions_path = '../disease_spread/data/interventions.csv'
    
    # Pick counties with 20 most cases:
    df_cases = pd.read_csv(cases_path)
    df_deaths = pd.read_csv(deaths_path)
    interventions = pd.read_csv(interventions_path)

    headers = df_cases.columns.values
    last_day = headers[-1]
    observed_days = len(headers[2:])
    
    df_cases = df_cases.sort_values(by=[last_day], ascending=False)
    df_cases = df_cases.iloc[:num_counties].copy()
    df_cases = df_cases.reset_index(drop=True)


    counties = df_cases['Combined_Key'].to_list()
    print(f'Order of M: {counties}')
    
    fips_list = df_cases['FIPS'].tolist()
    merge_df = pd.DataFrame({'merge':fips_list})
    df_deaths = df_deaths.loc[df_deaths['FIPS'].isin(fips_list)]

    #Select the 20 counties in the same order from the deaths dataframe by merging
    df_deaths = pd.merge(merge_df, df_deaths, left_on='merge', right_on='FIPS', how='outer')
    df_deaths = df_deaths.reset_index(drop=True)

    interventions = interventions[interventions['FIPS'] % 1000 != 0]
    id_cols = ['FIPS', 'STATE', 'AREA_NAME']
    int_cols = [col for col in interventions.columns.tolist() if col not in id_cols]
    interventions.fillna(1, inplace=True) ### to prevent NaN error
    for col in int_cols:
        interventions[col] = interventions[col].apply(lambda x: datetime.date.fromordinal(int(x))) ##changing date format
    interventions = interventions.loc[interventions['FIPS'].isin(fips_list)] ### filtering out top counties
    interventions = pd.merge(merge_df, interventions, left_on='merge', right_on='FIPS', how='outer')
    interventions = interventions.reset_index(drop=True)
    interventions = interventions.drop(['merge'], axis=1)
    interventions.drop(id_cols, axis=1, inplace=True)
    interventions_colnames = interventions.columns.values

    counter_list = []
    for index, row in df_cases.iterrows():
        cases_list = row[2:]
        counter = 1
        for cases in cases_list:
            if cases == 0:
                counter += 1
        counter_list.append(counter)

    df_cases = df_cases.drop(['FIPS', 'Combined_Key'], axis=1)
    df_cases = df_cases.T
    df_cases_dates = np.array(df_cases.index)
    df_cases = df_cases.to_numpy()
    
    df_deaths = df_deaths.drop(['merge','FIPS', 'Combined_Key'], axis=1)
    df_deaths = df_deaths.T
    df_deaths_dates = np.array(df_deaths.index)
    df_deaths = df_deaths.to_numpy()

    covariates1 = interventions.to_numpy()

    index = np.argmax(df_cases > 0, axis=0)
    cum_sum = np.cumsum(df_deaths, axis=0) >= 10
    index1 = np.where(np.argmax(cum_sum, axis=0) != 0, np.argmax(cum_sum, axis=0), cum_sum.shape[0])
    index2 = index1 - 30

    covariate1 = []
    covariate2 = []
    covariate3 = []
    covariate4 = []
    covariate5 = []
    covariate6 = []
    covariate7 = []

    for i in range(len(fips_list)):
        i2 = index2[i]
        req_dates = df_cases_dates[i2:]
        covariates2 = []
        req_dates = np.array(
            [datetime.datetime.strptime(x, '%m/%d/%y').date() for x in req_dates])  ##first case for each county

        for col in range(covariates1.shape[1]):
            covariates2.append(np.where(req_dates >= covariates1[i, col], 1, 0))
        covariates2 = np.array(covariates2).T

        N = len(req_dates)
        N2 = 50  ##from paper
        forecast = N2 - N

        if forecast < 0:
            print("FIPS: ", fips_list[i], " N: ", N)
            print("Error!!!! N2 is increasing!")
            N2 = N
            forecast = N2 - N

        addlst = [covariates2[N - 1]] * (forecast)
        covariates2 = np.append(covariates2, addlst, axis=0)
        covariate1.append(covariates2[:, 0])  # stay at home
        covariate2.append(covariates2[:, 1])  # >50 gatherings
        covariate3.append(covariates2[:, 2])  # >500 gatherings
        covariate4.append(covariates2[:, 3])  # public scools
        covariate5.append(covariates2[:, 4])  # restaurant dine-in
        covariate6.append(covariates2[:, 5])  # entertainment/gym
        covariate7.append(covariates2[:, 6])  # federal guidelines
        # covariate8.append(covariates2[:, 6]) #foreign travel ban (excluded)

    # converting to numpy array
    covariate1 = np.array(covariate1).T
    covariate2 = np.array(covariate2).T
    covariate3 = np.array(covariate3).T
    covariate4 = np.array(covariate4).T
    covariate5 = np.array(covariate5).T
    covariate6 = np.array(covariate6).T
    covariate7 = np.array(covariate7).T

    # covariate2 = covariate7

    # covariate4 = np.where( (covariate1 + covariate3 + covariate5 + covariate6 + covariate7) >=1, 1, 0) #any intervention
    # covariate5 = covariate5
    # covariate6 = covariate6
    # covariate7 = 0 #models should take only one covariate

    final_dict = {}
    final_dict['M'] = num_counties
    final_dict['N0'] = 6
    final_dict['N'] = np.asarray(num_counties* [observed_days]).astype(np.int)
    final_dict['N2'] = observed_days
    final_dict['x'] = np.arange(1, observed_days + 1).astype(np.int)
    final_dict['cases'] = df_cases.astype(np.int)
    final_dict['deaths'] = df_deaths.astype(np.int)
    final_dict['EpidemicStart'] = np.asarray(counter_list).astype(np.int)
#    final_dict['p'] = len(interventions_colnames) - 1 ### not sure whether to subtract 1 or not
    final_dict['covariate1'] = covariate1
    final_dict['covariate2'] = covariate2
    final_dict['covariate3'] = covariate3
    final_dict['covariate4'] = covariate4
    final_dict['covariate5'] = covariate5
    final_dict['covariate6'] = covariate6
    final_dict['covariate7'] = covariate7
    return final_dict

            
    
if __name__ == '__main__':
    #pick 20 counties
    #get_stan_parameters_our(20)
    get_stan_parameters()

