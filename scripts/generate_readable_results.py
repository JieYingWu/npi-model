import os
from os.path import join, exists
import pandas as pd
import numpy as np
import argparse
import re
import datetime as dt
import json
from pyperclip import copy

def to_latex(df):
  out = []
  out.append(' & '.join(df.keys()) + r' \\')
  for i in range(df.shape[0]):
    out.append(' & '.join(map(str, df.iloc[i,:])) + r' \\')
  return '\n'.join(out)
  

def main(result_dir=None, end_date='5/28', data_dir='data/us_data'):
  counties_path = join(data_dir, 'counties.csv')
  clustering_path = join(data_dir, 'clustering.csv')
  cases_path = join(data_dir, 'infections_timeseries.csv')
  deaths_path = join(data_dir, 'deaths_timeseries.csv')  
  start_dates_path = join(result_dir, 'start_dates.csv')
  geocode_path = join(result_dir, 'geocode.csv')
  summary_path = join(result_dir, 'summary.csv')

  supercounties_path = join(result_dir, 'supercounties.json')
  if not exists(supercounties_path):
    supercounties_path = join(data_dir, 'supercounties.json')
    print(f'WARNING: using {supercounties_path}, possibly old data')

  for path in [counties_path, clustering_path, cases_path, deaths_path,
               start_dates_path, geocode_path, summary_path]:
    if not exists(path):
      raise FileNotFoundError(f'not found: {path}')

  with open(supercounties_path, 'r') as file:
    supercounties = json.load(file)

  dtype = {'FIPS': str}
  counties = pd.read_csv(counties_path, dtype=dtype, delimiter=',')
  counties = counties.set_index('FIPS')

  populations = dict(zip(counties.index, counties['POP_ESTIMATE_2018']))
  for supercounty, fips_codes in supercounties.items():
    populations[supercounty] = sum(populations[fips] for fips in fips_codes)

  # new york county population, aggregated over manhattan, the bronx, brooklyn, queens, and staten island, resp.
  populations['36061'] = sum(populations[k] for k in ['36061', '36005', '36047', '36081', '36085'])
    
  converters = {'FIPS': lambda x : str(x).zfill(5)}
  cases = pd.read_csv(cases_path, converters=converters)
  cases = cases.set_index('FIPS')
  deaths = pd.read_csv(deaths_path, converters=converters)
  deaths = deaths.set_index('FIPS')
  supercounty_cases = []
  supercounty_deaths = []
  supercounties_index = []
  for supercounty, fips_codes in supercounties.items():
    state = cases.loc[fips_codes[0], 'Combined_Key'].split('-')[1].strip()
    cluster = supercounty.split('_')[-1]
    name = f'{state} Super-county Cluster {cluster}'

    supercounties_index.append(supercounty)
    supercounty_cases.append([name] + list(cases.loc[fips_codes].iloc[:, 1:].sum(0)))
    supercounty_deaths.append([name] + list(deaths.loc[fips_codes].iloc[:, 1:].sum(0)))

  supercounty_cases = pd.DataFrame(supercounty_cases, columns=cases.columns, index=supercounties_index)
  supercounty_deaths = pd.DataFrame(supercounty_deaths, columns=deaths.columns, index=supercounties_index)
  cases = cases.append(supercounty_cases)
  deaths = deaths.append(supercounty_deaths)
  
  clustering = pd.read_csv(clustering_path, dtype=dtype)
  clustering = dict(zip(clustering['FIPS'], clustering['cluster']))
  start_dates = list(pd.read_csv(start_dates_path, index_col=0).iloc[0])
  geocodes = list(pd.read_csv(geocode_path, index_col=0, dtype=str).iloc[0])
  summary = pd.read_csv(summary_path, index_col=0)
  indices = list(summary.index)

  # first, report and save the alpha values:
  alpha_indices = list(filter(lambda x : re.match(r'alpha\[\d\]', x) is not None, indices))
  alphas = summary.loc[alpha_indices, ['mean', 'sd']]
  alphas['NPI'] = ['$I_1$: Stay at home',
                   '$I_2$: >50 gathering',
                   '$I_3$: >500 gathering',
                   '$I_4$: Public schools',
                   '$I_5$: Restaurant dine-in',
                   '$I_6$: Entertainment/gym',
                   '$I_7$: Federal guidelines',
                   '$I_8$: Foreign travel ban',
                   '$I_9$: Mask mandate',
                   '$I_{10}:$',
                   '$I_{11}:$',
                   '$I_{12}:$',
                   '$I_{13}:$',
                   '$I_{14}:$'][:alphas.shape[0]]
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
    if fips not in supercounties:
      cluster = clustering[fips]
      row['County'] = '{Area_Name}, {State}'.format(fips=fips, **counties.loc[fips])
    else:
      cluster = fips.split('_')[-1]
      state_fips = fips.split('_')[0]
      row['County'] = '{State} Super-county Cluster {cluster}'.format(fips=fips, cluster=cluster, **counties.loc[state_fips])

    row['Cluster'] = str(int(cluster) + 1)
    row['R_0 (std)'] = '{mean:.03f} ({sd:.03f})'.format(**summary.loc[f'Rt_adj[1,{i + 1}]'])
    end_date_idx = dt.date(2020, *map(int, end_date.split('/'))).toordinal() - dt.date(2020, *map(int, start_dates[i].split('/')[:2])).toordinal()
    row[f'R_{end_date} (std)'] = '{mean:.03f} ({sd:.03f})'.format(**summary.loc[f'Rt_adj[{max(1, end_date_idx + 1)},{i + 1}]'])

    # get the number of predicted cases
    prediction_indices = list(filter(lambda x : re.match(r'prediction\[\d+,' + str(i + 1) + r'\]', x) is not None, indices))
    pred_cases = sum(list(summary.loc[prediction_indices, 'mean'])[:end_date_idx + 1])
    row['# (\\%) infected as predicted'] = '{:,d} ({:.01f})'.format(int(pred_cases), 999999999 if fips not in populations else pred_cases / populations[fips] * 100)

    if re.match(r'^\d{5}$', fips) is not None:
      num_cases = cases.loc[fips][-1]
      row['Measured cases'] = f'{num_cases:,d}'
      row['Fatality rate (measured death/cases)'] = f'{deaths.loc[fips][-1] / num_cases * 100:.02f}\\%'
      readable_summary.append(row)

    else:
      supercounty = fips
      for fips in supercounties[supercounty]:
        row = {}
        cluster = clustering[fips]
        row['County'] = '{Area_Name}, {State}'.format(fips=fips, **counties.loc[fips])
        row['Cluster'] = str(int(cluster) + 1)

        row['R_0 (std)'] = '{mean:.03f} ({sd:.03f})'.format(**summary.loc[f'Rt_adj[1,{i + 1}]'])
        end_date_idx = dt.date(2020, *map(int, end_date.split('/'))).toordinal() - dt.date(2020, *map(int, start_dates[i].split('/')[:2])).toordinal()
        row[f'R_{end_date} (std)'] = '{mean:.03f} ({sd:.03f})'.format(**summary.loc[f'Rt_adj[{end_date_idx + 1},{i + 1}]'])

        # get proportion of deaths in this county to deaths in supercounty, assume cases are proportional, and divide by population of the county
        supercounty_conversion_factor = deaths.loc[fips][-1] / deaths.loc[supercounty][-1]
        prediction_indices = list(filter(lambda x : re.match(r'prediction\[\d+,' + str(i + 1) + r'\]', x) is not None, indices))
        pred_cases = sum(list(summary.loc[prediction_indices, 'mean'])[:end_date_idx + 1]) * supercounty_conversion_factor
        row['# (\\%) infected as predicted'] = '{:,d} ({:.01f})'.format(int(pred_cases), pred_cases / populations[fips] * 100)          
        num_cases = cases.loc[fips][-1]
        row['Measured cases'] = f'{num_cases:,d}'
        row['Fatality rate (measured death/cases)'] = f'{deaths.loc[fips][-1] / num_cases * 100:.02f}\\%'
        readable_summary.append(row)

  readable_summary = pd.DataFrame(readable_summary)
  # print(readable_summary)
  latex_summary = to_latex(readable_summary)
  print(latex_summary)
  with open(join(result_dir, 'readable_summary.tex'), 'w') as file:
    file.write(latex_summary)
  
  readable_summary.to_csv(join(result_dir, 'readable_summary.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('result_dir', help='path to result dir to load the summary.csv, and to save the readable output version')
    parser.add_argument('--end_date', default='5/28', help='month/day (2020) to end predictions on')
    args = parser.parse_args()

    main(**args.__dict__)
