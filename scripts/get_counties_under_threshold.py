import json
import pandas as pd
from os.path import join, exists
from urllib.request import urlopen
import seaborn as sns
import re
from glob import glob
import argparse
import numpy as np
import datetime as dt



def get_fips_codes(result_dir):
  fips_codes = pd.read_csv(join(result_dir, 'geocode.csv'), dtype=str, index_col=0)
  return list(fips_codes.iloc[0, :])


def get_reproductive_ratio(result_dir):
  summary = pd.read_csv(join(result_dir, 'summary.csv'), index_col=0)
  indices = [x for x in summary.index if re.match(r'Rt_adj\[\d+,\d+\]', x) is not None]
  m = re.match(r'Rt_adj\[(?P<num_days>\d+),(?P<num_counties>\d+)\]', indices[-1])
  num_days = int(m.group('num_days'))
  num_counties = int(m.group('num_counties'))
  
  reproductive_ratio = summary.loc[indices, 'mean'].to_numpy()
  reproductive_ratio = reproductive_ratio.reshape(num_counties, num_days)
  # print(reproductive_ratio.shape, reproductive_ratio[0])
  return reproductive_ratio


def get_supercounties(result_dir):
  with open(join(result_dir, 'supercounties.json')) as file:
    supercounties = json.load(file)
  return supercounties


def is_county(fips):
  if not isinstance(fips, str):
    fips = str(fips).zfill(5)
  return re.match('\d{5}(?!_\d+)', fips) is not None


def is_result_dir(result_dir):
  return exists(join(result_dir, 'summary.csv'))


def get_start_dates(result_dir):
  start_dates_path = join(result_dir, 'start_dates.csv')
  start_dates = list(map(lambda x : dt.datetime.strptime(x, '%m/%d/%y'),
                         pd.read_csv(start_dates_path, index_col=0).iloc[0]))
  return start_dates
  

def get_counties_under_threshold(result_dirs, end_date, threshold=1.0, any_day=True):
  """Count the counties with Rt < threshold.

  If day is None, count the counties that lowered it at any point. Else, look at just that day. -1
  is the most recent day.

  :param result_dirs: 
  :returns: 
  :rtype:

  """
  count = 0
  total = 0
  for result_dir in result_dirs:
    fips_codes = get_fips_codes(result_dir)
    supercounties = get_supercounties(result_dir)
    total += sum(1 if is_county(fips) else len(supercounties[fips]) for fips in fips_codes)
    
    reproductive_ratio = get_reproductive_ratio(result_dir)
    assert reproductive_ratio.shape[0] == len(fips_codes)
    
    end_date_indices = [(end_date - sd).days for sd in get_start_dates(result_dir)]
    under = reproductive_ratio < threshold
    if any_day:
      under = np.array([np.any(r[:end_date_indices[i]]) for i, r in enumerate(under)])
    else:
      under = under[range(under.shape[0]), end_date_indices]

    under = under.astype(int)
    count += sum(under[i] * (1 if is_county(fips) else len(supercounties[fips])) for i, fips in enumerate(fips_codes))

  return count, total


def main(*, result_dir, data_dir, threshold, end_date):
  result_dirs = []
  for i, r in enumerate(result_dir):
    if is_result_dir(r):
      result_dirs.append(r)
    else:
      result_dirs += [x for x in glob(join(r, '*/')) if is_result_dir(x)]

  end_date = dt.datetime.strptime(end_date, '%m/%d/%y')
  
  num_any_day, total = get_counties_under_threshold(result_dirs, end_date=end_date, threshold=threshold, any_day=True)
  num_last_day, total_ = get_counties_under_threshold(result_dirs, end_date=end_date, threshold=threshold, any_day=False)
  assert total == total_
  print(f'{num_any_day} / {total} ({num_any_day / total * 100:.02f}%) had Rt < {threshold} at some point.')
  print(f'{num_last_day} / {total} ({num_last_day / total * 100:.02f}%) had Rt < {threshold} as of most recent data.')
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('result_dir', nargs='+', type=str,
                      help='path to result dir to load the summary.csv, or a dir containing several such dirs')
  parser.add_argument('--data_dir', default='./data', type=str,
                      help='path to result dir to load the summary.csv, or a dir containing several such dirs')
  parser.add_argument('--threshold', default=1.0, type=float, help='Rt threshold to count for')
  parser.add_argument('--end-date', default='8/2/20', help='end date')
  
  args = parser.parse_args()
  main(**args.__dict__)
