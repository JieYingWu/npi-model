import pandas as pd
from os.path import join, exists
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from math import isnan
import plotly.express as px
import plotly.graph_objects as go

visualizations_dir = './results/plots/states_r'


def plot_states(df,date):
    fig = go.Figure(data=go.Choropleth(
      locations=df['Combined_Key'],  # Spatial coordinates
      z=df['4/7/2020'].astype(float),  # Data to be color-coded
      locationmode='USA-states',  # set of locations match entries in `locations`
      colorscale='blues',
      marker_line_color='white'
    ))

    fig.update_layout(
      title_text='Rt',
      geo_scope='usa',  # limite map scope to USA
    )

    fig.layout.template = None
    fig.show()
    return fig


def main():
    data_dir = './data'
    filename = join(visualizations_dir, "states.png")
    #fips, beds = read_beds(data_dir)
    # insert reading from df
    df = pd.read_csv('../data/us_data/Dt_data_states.csv', delimiter=';')
    date = '4/7/2020'
    fig = plot_states(df, date)
    fig.write_image(filename)


if __name__ == '__main__':
    main()
