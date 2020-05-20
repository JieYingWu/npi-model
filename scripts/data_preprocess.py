import csv
import numpy as np
import pandas as pd
import datetime as dt
from os.path import join, exists
import json

pd.set_option('mode.chained_assignment', None)

def remove_negative_regions(df_cases, df_deaths, idx):
    """"
    Returns:
        df_cases: Infections time series with no negative values
        df_deaths: Deaths time series with no negative values
    """
    # drop if daily count negative
    sanity_check = df_cases.iloc[:, idx:].apply(lambda x: np.sum(x < 0), axis=1)
    drop_counties = sanity_check[sanity_check != 0].index
    df_cases = df_cases.drop(drop_counties)

    sanity_check2 = df_deaths.iloc[:, idx:].apply(lambda x: np.sum(x < 0), axis=1)
    drop_counties = sanity_check2[sanity_check2 != 0].index
    df_deaths = df_deaths.drop(drop_counties)

    # filter only the FIPS that are present in both cases and deaths timeseries
    intersect = list(set(df_cases['FIPS']) & set(df_deaths['FIPS']))
    df_cases = df_cases[df_cases['FIPS'].isin(intersect)]
    df_deaths = df_deaths[df_deaths['FIPS'].isin(intersect)]

    return df_cases, df_deaths

def select_top_regions(df_cases, df_deaths, interventions, num_counties, population, validation=False, supercounties=False, threshold=50):
    """"
    Returns:
        df_cases: Infections timeseries for top N places
        df_deaths: Death timeseries for top N places
        interventions: Intervention starting dates for top N places
        fips_list: FIPS of top N places
    """

    headers = df_cases.columns.values
    last_day = headers[-5]
    observed_days = len(headers[2:])

    if threshold is None:
        df_deaths = df_deaths.sort_values(by=[last_day], ascending=False)
        df_deaths = df_deaths.iloc[:num_counties].copy()
        df_deaths = df_deaths.reset_index(drop=True)
    else:
        cumulative_deaths = df_deaths.iloc[:, 2:].sum(axis=1).to_numpy()
        df_deaths = df_deaths.iloc[cumulative_deaths > threshold].copy()
        df_deaths = df_deaths.reset_index(drop=True)

    fips_list = df_deaths['FIPS'].tolist()

    merge_df = pd.DataFrame({'merge': fips_list})
    df_cases = df_cases.loc[df_cases['FIPS'].isin(fips_list)]
    # Select the 20 counties in the same order from the deaths dataframe by merging
    df_cases = pd.merge(merge_df, df_cases, left_on='merge', right_on='FIPS', how='outer')
    df_cases = df_cases.reset_index(drop=True)

    interventions = interventions.loc[interventions['FIPS'].isin(fips_list)]
    interventions = pd.merge(merge_df, interventions, left_on='merge', right_on='FIPS', how='outer')
    interventions = interventions.reset_index(drop=True)
    # interventions.fillna(dt.date.today().toordinal() + 365)

    population = population.loc[population['FIPS'].isin(fips_list)]
    population = pd.merge(merge_df, population, left_on='merge', right_on='FIPS', how='outer')
    population = population.reset_index(drop=True)
    
    df_cases.drop(['merge'], axis=1, inplace=True)
    interventions.drop(['merge'], axis=1, inplace=True)
    population.drop(['merge'], axis=1, inplace=True)

    #if validation > 0:
     #   df_cases = df_cases.iloc[:,:-(validation-1)]
     #   df_cases_val = df_cases.iloc[:,-(validation+1):]
     #   
     #   df_deaths = df_deaths.iloc[:,:-(validation-1)]
     #   df_deaths_val = df_deaths.iloc[:,-(validation+1):]

      #  return df_cases, df_deaths, interventions, population, fips_list
    return df_cases, df_deaths, interventions, population, fips_list


def merge_supercounties(cases, deaths, interventions, population, threshold=5, clustering=None, save_supercounties=False):
    """Join counties in the same state if they don't have enough deaths.

    Checks that the average daily deaths for the past 10 days is more than `threshold`. If the
    daily deaths are zero, completely drops the county.

    :param cases: 
    :param deaths: 
    :param interventions: 
    :param population: 
    :param threshold: 
    :param clustering: dictionary mapping [str] fips to [int] cluster label. 
      If None, supercounties are not limited to a single cluster.
    :returns: 
    :rtype: 

    """
    new_cases = []              # list of dictionaries, serving as rows
    new_deaths = []
    new_interventions = []
    new_population = []

    supercounty_to_cases_idx = {}         # map state fips strings to index in the above rows, for that supercounty
    supercounty_to_deaths_idx = {}
    supercounty_to_interventions_idx = {}
    supercounty_to_population_idx = {}
    supercounties = {}  # map STATEFIPS_CLUSTER to list of FIPS for counties in that supercounty
    for i, deaths_row in deaths.iterrows():
        fips = deaths_row['FIPS']
        fips_key = str(fips).zfill(5)
        cluster = clustering.get(fips_key)
        if cluster is None:
            continue
        
        cases_row = cases.loc[cases['FIPS'] == fips].copy().iloc[0]
        interventions_row = interventions.loc[interventions['FIPS'] == fips].copy().iloc[0]
        population_row = population.loc[population['FIPS'] == fips].copy().iloc[0]
        # print('CASES:\n', cases_row)
        # print('INTERVENTIONS:\n', interventions_row)
        
        county_deaths = deaths_row[2:].to_numpy()
        if np.all(county_deaths == 0):
            # county has no data to contribute at all
            continue
        if county_deaths[-5:].mean() >= threshold:
            # county should have enough data on its own
            new_cases.append(cases_row)
            new_deaths.append(deaths_row)
            new_interventions.append(interventions_row)
            new_population.append(population_row)
            continue

        supercounty = f'{fips_key[:2]}000_{cluster}'
        supercounties[supercounty] = supercounties.get(supercounty, []) + [fips_key]
        state_name = deaths_row['Combined_Key'].split('-')[1].strip()
        if cluster is None:
            area_name = f'{state_name} Supercounty'
        else:
            area_name = f'{state_name} Supercounty, Cluster {cluster}'

        # going to be adding to supercounty, so get rid of identifying info
        cases_row['FIPS'] = supercounty
        cases_row['Combined_Key'] = area_name
        deaths_row['FIPS'] = supercounty
        deaths_row['Combined_Key'] = area_name
        interventions_row['FIPS'] = supercounty
        interventions_row['AREA_NAME'] = area_name
        population_row['FIPS'] = supercounty
        
        if supercounty_to_cases_idx.get(supercounty) is None:
            print(f'added supercounty for {supercounty}')
            # first county encountered in state, just continue
            supercounty_to_cases_idx[supercounty] = len(new_cases)
            new_cases.append(cases_row)
            
            supercounty_to_deaths_idx[supercounty] = len(new_deaths)
            new_deaths.append(deaths_row)
            
            supercounty_to_population_idx[supercounty] = len(new_population)
            new_population.append(population_row)

            supercounty_to_interventions_idx[supercounty] = len(new_interventions)
            new_interventions.append(interventions_row)
            continue

        interventions_idx = supercounty_to_interventions_idx.get(supercounty)
        if np.any(interventions_row[2:] != new_interventions[interventions_idx][2:]):
            print(f"WARNING: couldn't merge {fips} with {supercounty} due to non-matching interventions")
            continue

        # encountered supercounty for which there is existing record and the interventions match, so merge all teh other info
        cases_idx = supercounty_to_cases_idx.get(supercounty)
        new_cases[cases_idx][2:] += cases_row[2:]
        
        deaths_idx = supercounty_to_deaths_idx.get(supercounty)
        new_deaths[deaths_idx][2:] += deaths_row[2:]
                
        population_idx = supercounty_to_population_idx.get(supercounty)
        new_population[population_idx][1] += population_row[1]
        print(f'MERGED {fips} with supercounty for {supercounty}')

    cases = pd.DataFrame(new_cases)
    deaths = pd.DataFrame(new_deaths)
    interventions = pd.DataFrame(new_interventions)
    population = pd.DataFrame(new_population)
    # print('CASES', cases, sep='\n')
    # print('DEATHS', deaths, sep='\n')
    # print('INTERVENTIONS', interventions, sep='\n')
    # print('POPULATION', population, sep='\n')

    print('supercounties:', supercounties)
    if save_supercounties:
        with open(join('data', 'us_data', 'supercounties.json'), 'w') as file:
            json.dump(supercounties, file)
            print('saved supercounties.json to data/us_data')
    return cases, deaths, interventions, population
    

def select_regions(cases, deaths, interventions, M, population, fips_list=None,
                   supercounties=True, clustering=None):
    """"
    Returns:
        df_cases: Infections timeseries for given fips
        df_deaths: Death timeseries for given fips
        interventions: Intervention starting dates for given fips
    """

    if fips_list is not None:
        cases = cases.loc[cases['FIPS'].isin(fips_list)]
        deaths = deaths.loc[deaths['FIPS'].isin(fips_list)]
        interventions = interventions.loc[interventions['FIPS'].isin(fips_list)]
        population = population.loc[population['FIPS'].isin(fips_list)]

    if supercounties:
        # join counties with less than a given threshold of deaths with other counties in the same state.
        # and if their interventions are the same as the other counties
        cases, deaths, interventions, population = merge_supercounties(
            cases, deaths, interventions, population, clustering=clustering, save_supercounties=(fips_list is None))

    return cases, deaths, interventions, population


def impute(df, allow_decrease_towards_end=True):
    """
    Impute the dataframe directly via linear interpolation

    Arguments:
    - df : pandas DataFrame

    Returns:
    - imputes pandas DataFrame

    """
    FIPS_EXISTS = False
    COMBINED_KEY_EXISTS = False
    
    if 'FIPS' in df:
        fips = df['FIPS']
        fips = fips.reset_index(drop=True)
        df = df.drop('FIPS', axis=1)
        FIPS_EXISTS = True
    if 'Combined_Key' in df:
        combined_key = df ['Combined_Key']
        combined_key = combined_key.reset_index(drop=True)
        df = df.drop('Combined_Key', axis=1)
        COMBINED_KEY_EXISTS = True

    header = df.columns.values
    df = df.to_numpy()
    
    for region in df:
        change_list = []
        # get first date of cases/deaths and skip it
        first = np.nonzero(region)[0]
        if len(first) == 0:
            continue
        change_list.append(first[0])
        for i, cell in enumerate(region[1:], 1):
            if i < first[0]:
                continue

            if i not in change_list:
                change_list.append(i)


            if cell < 0:
                region[i] = 0

            # Special Case where series is decreasing towards the end
            if i == (len(region)-1) and len(change_list) > 1 and not allow_decrease_towards_end:
                first_idx = change_list[0]
                diff = region[first_idx] - region[first_idx - 1]
                new_value = region[first_idx] + diff

                for j, idx in enumerate(change_list[1:], 1):
                    region[idx] = new_value
                    new_value += diff
                break

            if cell > region[change_list[0]]:
                if len(change_list) >= 3:
                    # cut first and last value of change list
                    first_, *change_list, last_ = change_list
                    region[change_list[0]:change_list[-1] + 1] = interpolate(change_list,
                                                                             region[first_],
                                                                             region[last_]) 
                    change_list = [last_]
                else:
                    change_list = [change_list[-1]]

    df = pd.DataFrame(df, columns=header)
    if COMBINED_KEY_EXISTS:
        df = pd.concat([combined_key, df], axis=1)
    if FIPS_EXISTS:
        df = pd.concat([fips, df], axis=1)
    return df

def interpolate(change_list, lower, upper):
    """
    Interpolate values with length ofchange_list between two given values lower and upper

    """

    x = np.arange(1, len(change_list) + 1)
    xp = np.array([0, len(change_list) + 1])
    fp = np.array([lower, upper])
    interpolated_values = np.interp(x, xp, fp)
    return np.ceil(interpolated_values)

def preprocessing_us_data(data_dir, mode='county'):
    """"
    Loads and cleans data
    Returns:
        df_cases: Infections timeseries based on daily count
        df_deaths: Deaths timeseries based on daily count
        interventions: Interventions data with dates converted to date format
    """
    assert mode in ['county', 'state'], ValueError()
    cases_path = join(data_dir, 'us_data/infections_timeseries_w_states.csv')
    deaths_path = join(data_dir, 'us_data/deaths_timeseries_w_states.csv')
        
    population_path = join(data_dir, 'us_data/counties.csv') #POP_ESTIMATE_2018     
    interventions_path = join(data_dir, 'us_data/interventions.csv')

    dtype = {'FIPS': str}
    df_cases = pd.read_csv(cases_path, dtype=dtype)
    df_deaths = pd.read_csv(deaths_path, dtype=dtype)
    interventions = pd.read_csv(interventions_path, dtype=dtype)
    counties = pd.read_csv(population_path, dtype=dtype)

    id_cols = ['FIPS', 'STATE', 'AREA_NAME']    
    int_cols = [col for col in interventions.columns.tolist() if col not in id_cols]

    interventions.drop([0], axis=0, inplace=True)
    interventions.fillna(1, inplace=True)

    for col in int_cols: ### convert date from given format
        interventions[col] = interventions[col].apply(lambda x: dt.date.fromordinal(int(x)))
    
    cols_population = ['FIPS', 'POP_ESTIMATE_2018']
    population = counties[cols_population]


    def get_daily_counts(L):
        diff = np.array([y - x for x, y in zip(L, L[1:])])
        L[1:] = diff
        return L

    #### get daily counts instead of cumulative
    df_cases.iloc[:, 2:] = df_cases.iloc[:, 2:].apply(get_daily_counts, axis=1)
    df_deaths.iloc[:, 2:] = df_deaths.iloc[:, 2:].apply(get_daily_counts, axis=1)

    return df_cases, df_deaths, interventions, population

def remove_negative_values(df):
    """ replaces all negative values with 0"""
    num = df._get_numeric_data()
    num[num < 0] = 0
    return df


def get_validation_dict(data_dir, cases, deaths, fips_list, cases_dates):
    """ get the dict with fips_codes as keys and list of days to hold out"""
    validation_days_path = join(data_dir, 'us_data', 'validation_days.csv')
    
    # set seed for numpy
    #np.random.seed(999)

    # get last day of current data
    first_day_cases = cases_dates[0]
    last_day_cases = cases_dates[-1]
    ordinal_last_day_cases = dt.datetime.strptime(last_day_cases, '%m/%d/%y').toordinal()
    ordinal_first_day_cases = dt.datetime.strptime(first_day_cases, '%m/%d/%y').toordinal()
    validation_days_dict  = {} 
   

    FLAG_NEW_DATA_AVAILABLE = False
    FLAG_NEW_FIPS = False
    # read the days ot be withheld if the file exists
    if exists(validation_days_path):
        with open(validation_days_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            dates = next(reader)
            first_date, last_date = dates
            ordinal_last_day = dt.datetime.strptime(first_date, '%m/%d/%y').toordinal()
            ordinal_first_day = dt.datetime.strptime(last_date, '%m/%d/%y').toordinal()
            if ordinal_last_day_cases > ordinal_last_day:
                FLAG_NEW_DATA_AVAILABLE = True
            next(reader)
            if not FLAG_NEW_DATA_AVAILABLE:
                for row in reader:
                    validation_days_dict[int(row[0])] = row[1:]
                
                not_included = []
                for j, i in enumerate(fips_list):
                    if i not in validation_days_dict.keys():
                        not_included.append((i,j))
                        FLAG_NEW_FIPS = True
    else:
        FLAG_NEW_DATA_AVAILABLE = True 

    if FLAG_NEW_DATA_AVAILABLE:
        validation_days_dict = {}
        for death, fips in zip(deaths.T, fips_list):
            for j in range(len(death)):
                if death[j] > 0:
                    validation_days_dict.setdefault(fips, []).append(j)
            validation_days_dict[fips] = get_unique_validation_days(validation_days_dict[fips])
        with open(validation_days_path, 'w', newline='' ) as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([first_day_cases, last_day_cases])
            writer.writerow(['FIPS', 'Indices of val days'])
            for key, val in validation_days_dict.items():
                list_to_write = [key] + val
                writer.writerow(list_to_write)
    

    if FLAG_NEW_FIPS:
        for fips, i in not_included:
            # i is the position of fips in fips_list
            for j in range(len(deaths.T[i])):
                if deaths.T[i,j] > 0:
                    validation_days_dict.setdefault(fips, []).append(j)
            validation_days_dict[fips] = get_unique_validation_days(validation_days_dict[fips])
        with open(validation_days_path, 'w', newline='' ) as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([first_day_cases, last_day_cases])
            writer.writerow(['FIPS', 'Indices of val days'])
            for key, val in validation_days_dict.items():
                list_to_write = [key] + val
                writer.writerow(list_to_write)

    return validation_days_dict
                    
def get_unique_validation_days(days_list):
    if len(days_list) < 3:
        return []
    
    np.random.seed(1234)
    
    num_validation_days = 3
    i = 0
    output = []
    while i < num_validation_days:
        day_to_drop = np.random.choice(days_list, 1, replace = True)[0]
        output.append(day_to_drop)
        idx = days_list.index(output[i])
        del days_list[idx]
        try:
            if abs(days_list[idx-1] - output[i]) < 2:
                del days_list[idx-1]
            if abs(days_list[idx+1] - output[i]) < 2:
                del days_list[idx+1]
        except IndexError:
            pass
        i += 1

    return output

        

def apply_validation(deaths, fips_list, validation_days_dict):
    """Sets the deaths of the days in the validation_days_dict to 0"""
    for i, fips in enumerate(fips_list):
      #j if len(validation_days_dict[fips]:   
        deaths[np.array(validation_days_dict[fips], dtype=np.int) , i] = 0
    return deaths
