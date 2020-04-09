import pandas as pd
from os.path import join, exists
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from math import isnan
import plotly.express as px
import plotly.graph_objects as go
import plotly as pl

def plot_states(df, date):
    fig = go.Figure(data=go.Choropleth(
        locations=df['Combined_Key'],  # Spatial coordinates
        z=df['4/7/2020'].astype(float) / 1e16,  # Data to be color-coded
        locationmode='USA-states',  # set of locations match entries in `locations`
        colorscale='blues',
        marker_line_color='white'
    ))

    fig.update_layout(
        title_text='Per State Reproduction Number R_t over time ' + str(date),
        geo_scope='usa',  # limite map scope to USA
    )
    # fig.show()
    return fig


def main():
    visualizations_dir = './results/plots/states_r/'
    df = pd.read_csv('../data/us_data/Dt_data_states.csv', delimiter=';')
    col_names = list(df.columns.values)[2:]
    date = '4/7/2020'
    for date in col_names:
        fig = plot_states(df, date)
        filename = visualizations_dir + date.replace("/", "")
        fig.write_image(filename)
        #pl.io.to_image(fig, format=None,scale=None, width=None, height=None)


if __name__ == '__main__':
    main()
