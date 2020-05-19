import pandas as pd
import numpy as np
from os.path import join, exists
import wget
import json


def to_fips(row):
    state = str(int(row['STATE'])).zfill(2)
    county = str(int(row['COUNTY'])).zfill(3)
    return state + county


def weighted_ifr_per_county():
    # used census data can be downloaded from
    # https://www2.census.gov/programs-surveys/popest/datasets/2010-2018/counties/asrh/
    census_data = pd.read_csv(join('data', 'us_data', 'cc-est2018-alldata.csv'), encoding="ISO-8859-1")
    census_data = census_data[census_data['YEAR'] == 11]
    ind = ['STATE', 'COUNTY', 'STNAME', 'CTYNAME', 'AGEGRP', 'TOT_POP']
    census_data = census_data[ind]

    # add rows for state level by aggregating the counties of each state
    states = np.unique(census_data['STATE'])
    state_names = np.unique(census_data['STNAME'])
    for state, state_name in zip(states, state_names):
        # add total population per age group over all counties of that state
        state_data = census_data[census_data['STATE'] == state]
        for age_group in range(19):
            total = state_data[state_data['AGEGRP'] == age_group]['TOT_POP'].sum()
            new_row = pd.Series([state, 0.0, state_name, 'NA', age_group, total], index=ind)
            census_data = census_data.append(new_row, ignore_index=True)

    census_data['FIPS'] = census_data.apply(to_fips, axis=1)
    census_data = census_data.sort_values(by=['FIPS', 'AGEGRP'])
    share = census_data.pivot_table(index=['FIPS'], columns=['AGEGRP'], values=['TOT_POP'])
    share = share.divide(share[('TOT_POP', 0)], axis=0)

    # add two age groups together to match the brackets in Verity et al.
    out = pd.DataFrame()
    out['0-9'] = share[[('TOT_POP', 1), ('TOT_POP', 2)]].sum(axis=1)
    out['10-19'] = share[[('TOT_POP', 3), ('TOT_POP', 4)]].sum(axis=1)
    out['20-29'] = share[[('TOT_POP', 5), ('TOT_POP', 6)]].sum(axis=1)
    out['30-39'] = share[[('TOT_POP', 7), ('TOT_POP', 8)]].sum(axis=1)
    out['40-49'] = share[[('TOT_POP', 9), ('TOT_POP', 10)]].sum(axis=1)
    out['50-59'] = share[[('TOT_POP', 11), ('TOT_POP', 12)]].sum(axis=1)
    out['60-69'] = share[[('TOT_POP', 13), ('TOT_POP', 14)]].sum(axis=1)
    out['70-79'] = share[[('TOT_POP', 15), ('TOT_POP', 16)]].sum(axis=1)
    out['80+'] = share[[('TOT_POP', 17), ('TOT_POP', 18)]].sum(axis=1)
    out['test'] = out.sum(axis=1)
    out.insert(0, 'FIPS', out.index)

    # this data is taken from Verity et al. 'Estimates of the severity of coronavirus disease 2019: a model-based analysis'
    ifr_unweighted = [0.0000161, 0.0000695, 0.000309, 0.000844, 0.00161, 0.00595, 0.0193, 0.0428, 0.0780]
    age_groups = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

    # county level demographics data
    out['fatality_rate'] = np.dot(out[age_groups], ifr_unweighted)

    out.to_csv(join('data', 'us_data', 'weighted_fatality.csv'), index=False)



def weighted_ifr_per_supercounty():
    # load supercounty info
    with open(join('data', 'us_data', 'supercounties.json')) as file:
        supercounties = json.loads(file.read().replace("'", '"'))

    # cluster_info = []
    # for i in range(num_supercounties):
    #     with open(join('data', 'us_data', 'supercounties_cluster=' + str(i) + '.json')) as f:
    #         data = f.read()
    #         data = data.replace('\'', '"')
    #         dict = json.loads(data)
    #         cluster_info.append(dict)

    # used census data can be downloaded from
    # https://www2.census.gov/programs-surveys/popest/datasets/2010-2018/counties/asrh/cc-est2018-alldata.csv
    fname = join('data', 'us_data', 'cc-est2018-alldata.csv')
    if not exists(fname):
        url = 'https://www2.census.gov/programs-surveys/popest/datasets/2010-2018/counties/asrh/cc-est2018-alldata.csv'
        wget.download(url, out=fname)
    census_data = pd.read_csv(fname, encoding="ISO-8859-1")
    census_data = census_data[census_data['YEAR'] == 11]
    ind = ['STATE', 'COUNTY', 'STNAME', 'CTYNAME', 'AGEGRP', 'TOT_POP']
    census_data = census_data[ind]

    # add rows for state level by aggregating the counties of each state
    states = np.unique(census_data['STATE'])
    state_names = np.unique(census_data['STNAME'])
    for state, state_name in zip(states, state_names):
        # add total population per age group over all counties of that state
        state_data = census_data[census_data['STATE'] == state]
        for age_group in range(19):
            total = state_data[state_data['AGEGRP'] == age_group]['TOT_POP'].sum()
            new_row = pd.Series([state, 0.0, state_name, 'NA', age_group, total], index=ind)
            census_data = census_data.append(new_row, ignore_index=True)

    ind = ['SUPERCOUNTY', 'STNAME', 'AGEGRP', 'TOT_POP']
    supercounty_ifr = pd.DataFrame(columns=ind)
    # add rows for supercounties by aggregating the counties per supercounty
    state_to_name = dict(zip(states, state_names))
    for supercounty, fips_codes in supercounties.items():
        print(f'{supercounty}: {fips_codes}')
        state = int(supercounty[:2])
        state_name = state_to_name[state]
        counties = [int(str(x).zfill(5)[2:]) for x in fips_codes]
        data = census_data[census_data['COUNTY'].isin(counties)]
        data = data[data['STATE'] == state]
        for age_group in range(19):
            total = data[data['AGEGRP'] == age_group]['TOT_POP'].sum()
            new_row = pd.Series([supercounty, state_name, age_group, total], index=ind)
            supercounty_ifr = supercounty_ifr.append(new_row, ignore_index=True)

    supercounty_ifr.sort_values(by=['SUPERCOUNTY', 'AGEGRP'])

    share = supercounty_ifr.pivot_table(index=['SUPERCOUNTY'], columns=['AGEGRP'], values=['TOT_POP'], aggfunc='first')
    share = share.divide(share[('TOT_POP', 0)], axis=0)

    # add two age groups together to match the brackets in Verity et al.
    out = pd.DataFrame()
    out['0-9'] = share[[('TOT_POP', 1), ('TOT_POP', 2)]].sum(axis=1)
    out['10-19'] = share[[('TOT_POP', 3), ('TOT_POP', 4)]].sum(axis=1)
    out['20-29'] = share[[('TOT_POP', 5), ('TOT_POP', 6)]].sum(axis=1)
    out['30-39'] = share[[('TOT_POP', 7), ('TOT_POP', 8)]].sum(axis=1)
    out['40-49'] = share[[('TOT_POP', 9), ('TOT_POP', 10)]].sum(axis=1)
    out['50-59'] = share[[('TOT_POP', 11), ('TOT_POP', 12)]].sum(axis=1)
    out['60-69'] = share[[('TOT_POP', 13), ('TOT_POP', 14)]].sum(axis=1)
    out['70-79'] = share[[('TOT_POP', 15), ('TOT_POP', 16)]].sum(axis=1)
    out['80+'] = share[[('TOT_POP', 17), ('TOT_POP', 18)]].sum(axis=1)
    out['test'] = out.sum(axis=1)
    out.insert(0, 'SUPERCOUNTY', out.index)

    # this data is taken from Verity et al. 'Estimates of the severity of coronavirus disease 2019: a model-based analysis'
    ifr_unweighted = [0.0000161, 0.0000695, 0.000309, 0.000844, 0.00161, 0.00595, 0.0193, 0.0428, 0.0780]
    age_groups = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

    # county level demographics data
    out['fatality_rate'] = np.dot(out[age_groups], ifr_unweighted)

    out.to_csv(join('data', 'us_data', 'weighted_fatality_supercounties.csv'), index=False)


if __name__ == '__main__':
    #weighted_ifr_per_county()
    weighted_ifr_per_supercounty()

