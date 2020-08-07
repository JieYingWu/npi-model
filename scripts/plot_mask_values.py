import json
import pandas as pd
from os.path import join, exists
from urllib.request import urlopen
import seaborn as sns
import wget
import re
from glob import glob
import argparse

import plotly.express as px
import plotly.graph_objects as go


fname = join('data', 'geojson-counties-fips.json')
if not exists(fname):
  print('downloading geojson data')
  url = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
  wget.download(url, out=fname)
  
with open(fname) as file:
  counties_geojson = json.load(file)
  
num_clusters = 5


def get_fips_codes(result_dir):
  fips_codes = pd.read_csv(join(result_dir, 'geocode.csv'), dtype=str, index_col=0)
  return list(fips_codes.iloc[0, :])


def get_alphas(result_dir):
  summary = pd.read_csv(join(result_dir, 'summary.csv'), index_col=0)
  indices = [x for x in summary.index if re.match(r'mask\[\d+\]', x) is not None]
  return list(summary.loc[indices, 'mean'])


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


def parse_mask_data(result_dir):
  """Parse the mask data from a single result dir, which contains summary.csv, supercounties.json, and geocode.csv.

  :param result_dirs: 
  :returns: dict mapping {fips -> mask value}
  :rtype: 

  """
  fips_codes = get_fips_codes(result_dir)
  alphas = get_alphas(result_dir)
  supercounties = get_supercounties(result_dir)
  out = {}
  for fips, alpha in zip(fips_codes, alphas):
    if is_county(fips):
      out[fips] = alpha
    else:
      for fips_code in supercounties[fips]:
        out[fips_code] = alpha
  return out


def get_mask_alphas(result_dirs):
  mask_data = {}
  for r in result_dirs:
    mask_data.update(parse_mask_data(r))
  return mask_data


def plot_mask_alphas(result_dirs):
  mask_alphas = get_mask_alphas(result_dirs)
  df = pd.DataFrame.from_dict(mask_alphas, orient='index', columns=['Alpha Value'])

  fig = px.choropleth(
    df,
    geojson=counties_geojson,
    locations=df.index,
    color='Alpha Value',
    color_continuous_scale='Reds',
    # range_color=()
  )
  fig.update_geos(fitbounds="locations", visible=False)
  fig.update_layout(
    coloraxis_showscale=True,
    font=dict(family='Helvetica'),
    coloraxis_colorbar=dict(
      title='',
      thicknessmode="pixels",
      thickness=10,
      lenmode='pixels',
      len=300,
      # ticks='outside',
      # tickvals=[0, 10, 20, 30, 40, 50],
      # ticktext=['0', '10', '20', '30', '40', '50+'],
      # dtick=5,
      # yanchor='middle'
    ))
  #fig.show()
  fig.write_image(join('visualizations', f'mask_alphas.pdf'), scale=3)
  

def main(result_dir):
  result_dirs = []
  for r in result_dir:
    if is_result_dir(r):
      result_dirs.append(r)
    else:
      result_dirs += [x for x in glob(join(r, '*/')) if is_result_dir(x)]

  plot_mask_alphas(result_dirs)
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('result_dir', nargs='+', type=str,
                      help='path to result dir to load the summary.csv, or a dir containing several such dirs')
  args = parser.parse_args()
  main(args.result_dir)
