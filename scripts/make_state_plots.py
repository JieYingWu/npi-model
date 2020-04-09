import pandas as pd
from os.path import join, exists
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from math import isnan
import plotly.express as px

visualizations_dir = './results/plots/states_r'
beds_key = "ICU Beds"
cmap = plt.get_cmap('Blues')


def is_county(fips):
    return fips[2:] != '000'


def is_state(fips):
    return fips[2:] == '000'


def read_beds(data_dir):
    filename = join(data_dir, 'counties.csv')
    df = pd.read_csv(filename, converters={
        "FIPS": str,
        # beds_key : lambda x : 0. if x == 'NA' else float(x)
    })
    fips_codes = []
    beds = []
    for fips, bs in zip(list(df['FIPS']), list(df[beds_key])):
        if is_county(fips) and bs != 'NA' and bs != '' and not isnan(float(bs)):
            fips_codes.append(fips)
            beds.append(int(bs))
    return fips_codes, beds


def plot_states(df):
    fig = go.Figure(data=go.Choropleth(
      locations=df['code'],  # Spatial coordinates
      z=df['total exports'].astype(float),  # Data to be color-coded
      locationmode='USA-states',  # set of locations match entries in `locations`
      colorscale='Reds',
      marker_line_color='white',
      title="Rt values",
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
    fig = plot_states(df)
    fig.write_image(filename)


if __name__ == '__main__':
    main()
