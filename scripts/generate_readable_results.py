import os
from os.path import join, exists
import pandas as pd
import numpy as np
import argparse
import re
import datetime as dt

def main(result_dir=None, end_date='5/28', data_dir='data/us_data'):
  counties_path = join(data_dir, 'counties.csv')
  clustering_path = join(data_dir, 'clustering.csv')
  cases_path = join(data_dir, 'infections_timeseries.csv')
  deaths_path = join(data_dir, 'deaths_timeseries.csv')  
  start_dates_path = join(result_dir, 'start_dates.csv')
  geocode_path = join(result_dir, 'geocode.csv')
  summary_path = join(result_dir, 'summary.csv')
  for path in [counties_path, clustering_path, cases_path, deaths_path,
               start_dates_path, geocode_path, summary_path]:
    if not exists(path):
      raise FileNotFoundError(f'not found: {path}')

  dtype = {'FIPS': str}
  counties = pd.read_csv(counties_path, dtype=dtype, delimiter=',')
  counties = counties.set_index('FIPS')
  
  populations = dict(zip(counties.index, counties['POP_ESTIMATE_2018']))

  converters = {'FIPS': lambda x : str(x).zfill(5)}
  cases = pd.read_csv(cases_path, converters=converters)
  cases = cases.set_index('FIPS')
  deaths = pd.read_csv(deaths_path, converters=converters)
  deaths = deaths.set_index('FIPS')
  print(cases)

  clustering = pd.read_csv(clustering_path, dtype=dtype)
  clustering = dict(zip(clustering['FIPS'], clustering['cluster']))
  start_dates = list(pd.read_csv(start_dates_path, index_col=0).iloc[0])
  geocodes = list(pd.read_csv(geocode_path, index_col=0, dtype=str).iloc[0])
  summary = pd.read_csv(summary_path, index_col=0)
  indices = list(summary.index)

  # first, report and save the alpha values:
  alpha_indices = list(filter(lambda x : re.match(r'alpha\[\d\]', x) is not None, indices))
  alphas = summary.loc[alpha_indices, ['mean', 'sd']]
  alphas['NPI'] = ['I1: Stay at home',
                   'I2: >50 gathering',
                   'I3: >500 gathering',
                   'I4: Public schools',
                   'I5: Restaurant dine-in',
                   'I6: Entertainment/gym',
                   'I7: Federal guidelines',
                   'I8: Foreign travel ban']
  alphas = alphas.set_index('NPI')
  print(alphas)
  print('================================================================================')
  print(f'alpha value [mean (std)] for copying:')
  print('================================================================================')
  print('\n'.join('{:.03f} ({:.03f})'.format(row['mean'], row['sd']) for i, row in alphas.iterrows()))
  print('================================================================================')
  alphas.to_csv(join(result_dir, 'alphas.csv'))

  readable_summary = []         # list of dicts, which are the rows
  print(*map(int, end_date.split('/')))
  for i, fips in enumerate(geocodes):
    row = {}
    is_county = re.match(r'\d{5}_\d', fips) is None
    if is_county:
      cluster = clustering[fips]
      row['County'] = '{fips}\n{Area_Name}, {State}'.format(fips=fips, **counties.loc[fips])
    else:
      cluster = fips.split('_')[-1]
      state_fips = fips.split('_')[0]
      row['County'] = '{fips}\n{State} Super-county Cluster {cluster}'.format(fips=fips, cluster=cluster, **counties.loc[state_fips])

    row['Cluster'] = cluster
    row['R_0 (std)'] = '{mean:.03f} ({sd:.03f})'.format(**summary.loc[f'Rt_adj[1,{i + 1}]'])
    end_date_idx = dt.date(2020, *map(int, end_date.split('/'))).toordinal() - dt.date(2020, *map(int, start_dates[i].split('/')[:2])).toordinal()
    row[f'R_{end_date} (std)'] = '{mean:.03f} ({sd:.03f})'.format(**summary.loc[f'Rt_adj[{end_date_idx + 1},{i + 1}]'])

    # get the number of predicted cases
    prediction_indices = list(filter(lambda x : re.match(r'prediction\[\d+,' + str(i + 1) + r'\]', x) is not None, indices))
    pred_cases = sum(list(summary.loc[prediction_indices, 'mean'])[:end_date_idx + 1])
    row['# (%) infected as predicted'] = '{:d} ({:.01f})'.format(int(pred_cases), 999999999 if fips not in populations else pred_cases / populations[fips] * 100)

    if is_county and fips in cases.index:
      num_cases = sum(cases.loc[fips][1:])
      row['Measured cases'] = num_cases
      row['Fatality rate (measured death/cases)'] = f'{sum(deaths.loc[fips][1:]) / num_cases}%'
    else:
      row['Measured cases'] = 99999999
      row['Fatality rate (measured death/cases)'] = 99999999
      
  
    # TODO: measured cases and fatality rate
    # row['Measured cases'] = 
    # row['R_0 (std)'] = summary.loc[f'Rt_adj[0, {i + 1}]']
    readable_summary.append(row)
  readable_summary = pd.DataFrame(readable_summary)
  print(readable_summary)
  readable_summary.to_csv(join(result_dir, 'readable_summary.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('result_dir', help='path to result dir to load the summary.csv, and to save the readable output version')
    parser.add_argument('--end_date', default='5/28', help='month/day (2020) to end predictions on')
    args = parser.parse_args()

    main(**args.__dict__)
