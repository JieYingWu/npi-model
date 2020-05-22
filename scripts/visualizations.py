import json
import pandas as pd
from os.path import join, exists
from urllib.request import urlopen
import seaborn as sns
import wget

import plotly.express as px
import plotly.graph_objects as go


fname = join('data', 'geojson-counties-fips.json')
if not exists(fname):
  print('downloading geojson data')
  url = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
  wget.download(url, out=fname)
  
with open(fname) as file:
  counties_geojson = json.load(file)
  

def load_supercounties():
  with open(join('data', 'us_data', 'supercounties.json'), 'r') as file:
    supercounties = json.load(file)
  return supercounties


def load_clustering():
  dtype = {'FIPS': str, 'cluster': str}
  return pd.read_csv(join('data', 'us_data', 'clustering.csv'), dtype=dtype)


def load_timeseries(timeseries_type):
  dtype = {'FIPS': str}
  return pd.read_csv(join('data', 'us_data', f'{timeseries_type}_timeseries.csv'), dtype=dtype)


def in_state(county, state):
  return county[:2] == state[:2]


def filter_by_state(df, state=None):
  if state is None:
    return df
  return df[[in_state(fips, state) for fips in df['FIPS']]]


num_clusters = 5
color_palette = sns.color_palette('Set1', n_colors=num_clusters)
color_palette = [f'#{int(255*t[0]):02x}{int(255*t[1]):02x}{int(255*t[2]):02x}' for t in color_palette]
color_discrete_map = dict((str(cluster), color_palette[cluster]) for cluster in range(num_clusters))
color_discrete_map['-1'] = '#ffffff'
height = 400


def plot_clustering(state=None):
  clustering = filter_by_state(load_clustering(), state)
  fig = px.choropleth(
    clustering,
    geojson=counties_geojson,
    locations='FIPS',
    color='cluster',
    color_discrete_map=color_discrete_map,
    scope='usa' if state is None else None
  )
  fig.update_layout(legend_title_text='Cluster Label',
                    legend=dict(traceorder='normal', orientation='h'),
                    font=dict(family='Times New Roman'))
  if state is None:
    fig.update_geos(visible=False)
  else:
    fig.update_geos(fitbounds="locations", visible=False)
    
  if state is None:
    fig.write_image(join('visualizations', f'us_clustering.pdf'))
  else:
    fig.write_image(join('visualizations', f'{state}_clustering.pdf'), scale=3)


def plot_deaths(state=None):
  deaths = filter_by_state(load_timeseries('deaths'), state)
  col = deaths.columns.tolist()[-1]
  fig = px.choropleth(
    deaths,
    geojson=counties_geojson,
    locations='FIPS',
    color=col,
    color_continuous_scale='Reds',
    range_color=(0, 50)
  )
  fig.update_geos(fitbounds="locations", visible=False)
  fig.update_layout(
    coloraxis_showscale=True,
    font=dict(family='Times New Roman'),
    coloraxis_colorbar=dict(
      title='',
      thicknessmode="pixels",
      thickness=10,
      lenmode='pixels',
      len=300,
      ticks='outside',
      tickvals=[0, 10, 20, 30, 40, 50],
      ticktext=['0', '10', '20', '30', '40', '50+'],
      dtick=5,
      yanchor='middle'))
  fig.write_image(join('visualizations', f'{state}_deaths.pdf'), scale=3)
  
  
def plot_supercounties(state=None, num_clusters=5):
  """Plot the supercounties for the given state.

  Plot each supercounty on its own, plus all the counties that are included but not a part of a supercounty.

  :param state: 
  :param num_clusters: 
  :returns: 
  :rtype: 

  """
  
  supercounties = load_supercounties()
  clustering = filter_by_state(load_clustering(), state)
  deaths = load_timeseries('deaths')
  for cluster in range(num_clusters):
    supercounty = f'{state}_{cluster}'
    counties = set(supercounties.get(supercounty, []))
    y = [c if fips in counties else '-1' for fips, c in zip(clustering['FIPS'], clustering['cluster'])]
    fig = px.choropleth(
      geojson=counties_geojson,
      locations=clustering['FIPS'],
      color=y,
      color_discrete_map=color_discrete_map,
    )
    fig.update_layout(showlegend=False, font=dict(family='Times New Roman'))
    fig.update_geos(fitbounds="locations", visible=False)
    fig.write_image(join('visualizations', f'{supercounty}_supercounty.pdf'), scale=3)


def make_plots(state=None):
  plot_deaths(state)
  plot_clustering(state)
  plot_supercounties(state)


if __name__ == '__main__':
  # make_plots('36000')           # new york
  # make_plots('48000')           # texas
  # make_plots('06000')           # california
  # make_plots('24000')           # maryland
  # make_plots('53000')           # washington
  make_plots()

