import os
import csv

import numpy as np
import future.backports.datetime as datetime
import pandas as pd


path = 'data/COVID-19-up-to-date.csv'
path_cases_new = 'data/COVID-19-up-to-date-cases-clean.csv'
path_deaths_new = 'data/COVID-19-up-to-date-deaths-clean.csv'
interventions = pd.read_csv('data/interventions.csv')

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


    """
    mod_interventions = pd.DataFrame(columns=['Country', 'school/uni closures', 'self-isolating if ill',
                                              'banning public events', 'any government intervention',
                                              'complete/partial lockdown', 'social distancing/isolation'])

    mod_interventions['Country'] = interventions.iloc[0:11]['Country']
    mod_interventions['school/uni closures'] = interventions.iloc[0:11]['schools_universities']
    mod_interventions['self-isolating if ill'] = interventions.iloc[0:11]['self_isolating_if_ill']
    mod_interventions['banning public events'] = interventions.iloc[0:11]['public_events']
    mod_interventions['social distancing/isolation'] = interventions.iloc[0:11]['social_distancing_encouraged']
    mod_interventions['complete/partial lockdown'] = interventions.iloc[0:11]['lockdown']

    for col in mod_interventions.columns.tolist():
        if col == 'Country' or col == 'complete/partial lockdown':
            continue
        col1 = pd.to_datetime(mod_interventions[col]).dt.date
        col2 = pd.to_datetime(mod_interventions['complete/partial lockdown']).dt.date
        mod_interventions[col] = np.where(col1 > col2, col2, col1).astype(str)

    mod_interventions.sort_values('Country', inplace=True)


    countries = sorted(['Denmark', 'Italy', 'Germany', 'Spain', 'United Kingdom', 'France', 'Norway', 'Belgium', 'Austria', 'Sweden', 'Switzerland'])
    print(countries)

    countries_list = []
    cases_dict = {}
    deaths_dict = {}
    cases_dict_padded = {}
    deaths_dict_padded = {}
    start_date_dict = {}
    start_date = datetime.date(2019,12,31)

    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)


        for row in reader:
            dateRep, day, month, year, cases, deaths, country, geoId,countryCode, pop = row
            current_date = datetime.date(int(year), int(month), int(day))
            if country not in countries:
                continue
            
            if cases != '0':
                start_date_dict[country] = current_date
            if country not in countries_list:
                countries_list.append(country)
                cases_dict[country] = []
                deaths_dict[country] = []
                cases_dict_padded[country] = {} 
                deaths_dict_padded[country] = {}
                if current_date != datetime.date(2019,12,31):
                    for i in range((current_date+datetime.timedelta(days=1)-start_date).days):
                        new_date = start_date + datetime.timedelta(days=i)
                        cases_dict_padded[country][str(new_date)] = 0
                        deaths_dict_padded[country][str(new_date)] = 0
            
            cases_dict_padded[country][str(current_date)] = cases
            deaths_dict_padded[country][str(current_date)] = deaths
            cases_dict[country].append(cases)
            deaths_dict[country].append(deaths)

    # check that all countries have the same number of dates
    len_list = []
    for key, value in cases_dict.items():
        len_list.append(len(value))

    start_date_list = []
    for key, value in start_date_dict.items():    
        start_date_list.append(value)

    #create final arrays:
    cases_list = []
    for idx, (country, cases_dict_list) in enumerate(cases_dict.items()):
        cases_list.append([])
        for cases in cases_dict_list:
            cases_list[idx].append(cases)

    deaths_list = []
    for idx, (country, deaths_dict_list) in enumerate(deaths_dict.items()):
        deaths_list.append([])
        for deaths in deaths_dict_list:
            deaths_list[idx].append(deaths)

    start_date_int_list = []
    for date in start_date_list:
        date = ((date+datetime.timedelta(days=1)-start_date).days)
        start_date_int_list.append(date)



    with open(path_cases_new, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for i, country in enumerate(countries_list):
            row = [country] + [cases for cases in reversed(cases_dict[country])]
            writer.writerow(row)

    with open(path_deaths_new, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for i, country in enumerate(countries_list):
            row = [country] + [deaths for deaths in reversed(deaths_dict[country])]
            writer.writerow(row)

    final_dict = {}
    final_dict['M'] = len(countries)
    #final_dict['N0'] = 
    final_dict['N'] = np.asarray(len_list).astype(np.int)
    final_dict['N2'] = len_list[0]
    final_dict['x'] = np.arange(1, (datetime.date(2020,3,28)+datetime.timedelta(days=1)-start_date).days + 1)
    final_dict['cases'] = np.asarray(cases_list).T.astype(np.int)
    final_dict['deaths'] = np.asarray(deaths_list).T.astype(np.int)
    final_dict['EpidemicStart'] = np.asarray(start_date_int_list).astype(np.int)
    final_dict['p'] = mod_interventions.to_numpy()
    return final_dict


def get_stan_parameters_our(num_counties):
    cases_path = '../disease_spread/data/infections_timeseries.csv'
    deaths_path = '../disease_spread/data/deaths_timeseries.csv'

    # Pick counties with 20 most cases:
    df_cases = pd.read_csv(cases_path)
    df_deaths = pd.read_csv(deaths_path)
    
    headers = df_cases.columns.values
    last_day = headers[-1]
    observed_days = len(headers[2:])
    
    df_cases = df_cases.sort_values(by=[last_day], ascending=False)
    df_cases = df_cases.iloc[:num_counties].copy()
    df_cases = df_cases.reset_index(drop=True)
    
    fips_list = df_cases['FIPS'].tolist()
    merge_df = pd.DataFrame({'merge':fips_list})
    df_deaths = df_deaths.loc[df_deaths['FIPS'].isin(fips_list)]

    df_deaths = pd.merge(merge_df, df_deaths, left_on='merge', right_on='FIPS', how='outer')
    df_deaths = df_deaths.reset_index(drop=True)
    
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
    df_cases = df_cases.to_numpy()
    
    df_deaths = df_deaths.drop(['merge','FIPS', 'Combined_Key'], axis=1)
    df_deaths = df_deaths.T
    df_deaths = df_deaths.to_numpy()

    
    final_dict = {}
    final_dict['M'] = num_counties
    # #final_dict['N0'] = 
    final_dict['N'] = np.asarray(num_counties* [observed_days]).astype(np.int)
    final_dict['N2'] = observed_days
    final_dict['x'] = np.arange(1, observed_days + 1).astype(np.int)
    final_dict['cases'] = df_cases.astype(np.int)
    final_dict['deaths'] = df_deaths.astype(np.int)
    final_dict['EpidemicStart'] = np.asarray(counter_list).astype(np.int)

    return final_dict

            
                


    
if __name__ == '__main__':
    #pick 20 counties
    get_stan_parameters_our(20)


