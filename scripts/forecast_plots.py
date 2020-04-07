from os.path import join, exists
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# change here variables for different plotting options
plot_settings = 'usa'  # choose 'eu' for europe and 'usa' for usa plots
base_model = True  # True for prediction/E_deaths, False for prediction0/E_deaths0
# to match with IC paper select base_model==True
last_day_to_plot = '03-31-2020'  # predict to this date


# saving some params for plot settings
if plot_settings == 'eu':
    start_day_of_confirmed = '12-31-2019'
    if base_model:
        results_folder = 'europe'
    else:
        results_folder = 'europe0'
elif plot_settings == 'usa':
    start_day_of_confirmed = '01-22-2020'
    if base_model:
        results_folder = 'usa'
    else:
        results_folder = 'usa0'


def plot_forecasts_wo_dates_quantiles(quantiles_dict, confirmed_cases, county_name, plot_choice,
                                      num_of_country, dict_of_start_dates, dict_of_eu_geog, save_image=True):
    '''
    :param quantiles_dict: stores values of quantiles
    :param confirmed_cases: real confirmed cases
    :param plot_choice: select between deaths/infections plot
    :param num_of_country: index of geography from dict_of_eu_geog
    :param dict_of_start_dates: each geography has different start day, its stored in the dict
    :param dict_of_eu_geog: dictionary of geography names
    :param save_image: True for save, False for not saving
    :return: beautiful magestic plot
    '''

    if plot_choice == 0:
        metric = "infections"
    elif plot_choice == 1:
        metric = "deaths"

    base = datetime.datetime.strptime(dict_of_start_dates[num_of_country], '%m-%d-%Y')
    days_to_predict = (datetime.datetime.strptime(last_day_to_plot, '%m-%d-%Y') - base).days
    print("Will make plot for {} days".format(days_to_predict))
    date_list = [base + datetime.timedelta(days=x) for x in range(days_to_predict)]

    # make the shapes match
    diff = days_to_predict - np.shape(confirmed_cases)[0]
    if diff <= 0:
        barplot_values = list(confirmed_cases[:days_to_predict])
    else:
        barplot_missing_values = np.zeros(days_to_predict - np.shape(confirmed_cases)[0])
        barplot_values = list(confirmed_cases) + list(barplot_missing_values)

    # plot creation
    fig = plt.figure('Forecast ')
    ax = fig.add_subplot(111)
    ax.fill_between(date_list, quantiles_dict['2.5%'], quantiles_dict['97.5%'], alpha=0.25, color='b')
    ax.fill_between(date_list, quantiles_dict['25%'], quantiles_dict['75%'], alpha=0.2, color='b')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.bar(date_list, barplot_values, color='r', width=0.9, alpha=0.3)
    ax.set_ylabel("Daily number of {}".format(metric))
    ax.set_xlabel("Date")

    if county_name == "":
        geography_name = str(dict_of_eu_geog[num_of_country])
    else:
        geography_name = str(county_name)
    ax.title.set_text(geography_name)
    ax.xaxis_date()
    fig.autofmt_xdate()

    if save_image:
        name = str(metric) + str(dict_of_eu_geog[num_of_country])
        plt.tight_layout()
        fig.savefig('../results/plots/{}/{}.jpg'.format(results_folder,name))
        fig.clf()
    else:
        plt.show()


def plot_daily_infections_num(path, confirmed_cases, county_name, plot_choice, num_of_country, dict_of_start_dates,
                              dict_of_eu_geog):
    print(county_name)
    # 1 for deaths; 0 for infections
    plot_name = ""
    if plot_choice == 0:
        plot_name += "prediction"
    elif plot_choice == 1:
        plot_name += "E_deaths"

    # True for prediction/E_deaths, False for prediction0/E_deaths0
    if not base_model:
        plot_name += "0["
    else:
        plot_name += "["

    base = datetime.datetime.strptime(dict_of_start_dates[num_of_country], '%m-%d-%Y')
    days_to_predict = (datetime.datetime.strptime(last_day_to_plot, '%m-%d-%Y') - base).days

    df = pd.read_csv(path, delimiter=';', index_col=0)
    row_names = list(df.index.tolist())
    list2_5, list25, list50, list75, list97_5 = [], [], [], [], []
    county_number = str(num_of_country + 1) + ']'

    for name in row_names:
        if plot_name in name:
            if name.split(",")[1] == county_number:
                rowData = df.loc[name, :]
                list2_5.append(rowData['2.5%'])
                list25.append(rowData['25%'])
                list50.append(rowData['50%'])
                list75.append(rowData['75%'])
                list97_5.append(rowData['97.5%'])

                # if last day of prediction was saved, exit
                if name.split(",")[0] == (plot_name + str(days_to_predict)):
                    break

    quantiles_dict = {'2.5%': list2_5,'25%': list25, '50%': list50, '75%': list75, '97.5%': list97_5}
    plot_forecasts_wo_dates_quantiles(quantiles_dict, confirmed_cases, county_name,
                                      plot_choice, num_of_country, dict_of_start_dates, dict_of_eu_geog)


def read_true_cases_europe(plot_choice, num_of_country, dict_of_start_dates, dict_of_eu_geog):
    '''
    1 for deaths forecast
    0 for infections forecast
    '''
    if plot_choice == 0:
        filepath = "../data/europe_data/COVID-19-up-to-date-cases-clean.csv"
    else:
        filepath = "../data/europe_data/COVID-19-up-to-date-deaths-clean.csv"

    df = pd.read_csv(filepath, delimiter=',', header=None)

    confirmed_start_date = datetime.datetime.strptime(start_day_of_confirmed, '%m-%d-%Y')
    forecast_start_date = datetime.datetime.strptime(dict_of_start_dates[num_of_country], '%m-%d-%Y')
    diff = (forecast_start_date - confirmed_start_date).days + 1
    confirmed_cases = list(df.iloc[num_of_country, diff:])

    return confirmed_cases


def read_true_cases_us(plot_choice, num_of_country, dict_of_start_dates, dict_of_eu_geog):
    # 1 for deaths forecast; 0 for infections forecast
    if plot_choice == 0:
        filepath = "../data/us_data/infections_timeseries.csv"
    else:
        filepath = "../data/us_data/deaths_timeseries.csv"

    df = pd.read_csv(filepath, delimiter=',', index_col=0)
    fips = dict_of_eu_geog[num_of_country]

    confirmed_start_date = datetime.datetime.strptime(start_day_of_confirmed, '%m-%d-%Y')
    forecast_start_date = datetime.datetime.strptime(dict_of_start_dates[num_of_country], '%m-%d-%Y')
    diff = (forecast_start_date - confirmed_start_date).days + 1  # since it also has a name skip it

    confirmed_cases = list(df.loc[fips][diff:])
    county_name = df.loc[fips][0]
    return confirmed_cases, county_name


# create a batch of all possible plots for usa
def make_all_us_plots():
    dict_of_start_dates = {0: '2-16-2020', 1: '2-26-2020', 2: '2-20-2020', 3: '2-18-2020', 4: '02-21-2020',
                           5: '2-21-2020', 6: '2-22-2020', 7: '2-17-2020', 8: '2-02-2020', 9: '02-18-2020',
                           10: '2-16-2020', 11: '2-27-2020', 12: '2-20-2020', 13: '2-21-2020', 14: '02-28-2020',
                           15: '2-22-2020', 16: '2-25-2020', 17: '2-25-2020', 18: '2-26-2020', 19: '2-27-2020'}

    dict_of_eu_geog = {0: 36061, 1: 36119, 2: 36059, 3: 36103, 4: 17031, 5: 26163, 6: 36087, 7: 34003,
                       8: 53033, 9: 6037, 10: 22071, 11: 36071, 12: 9001, 13: 34013, 14: 12086,
                       15: 26125, 16: 25017, 17: 34017, 18: 25025, 19: 34023}

    path = "../results/US_summary.csv"

    for plot_choice in range(0, 2):
        for num_of_country in dict_of_eu_geog.keys():
            confirmed_cases, county_name = read_true_cases_us(plot_choice, num_of_country, dict_of_start_dates,
                                                              dict_of_eu_geog)
            plot_daily_infections_num(path, confirmed_cases, county_name, plot_choice, num_of_country,
                                      dict_of_start_dates, dict_of_eu_geog)
    return


# create a batch of all possible plots for europe
def make_all_eu_plots():
    dict_of_start_dates = {0: '02-22-2020', 1: '02-18-2020', 2: '02-21-2020', 3: '02-07-2020',
                           4: '02-15-2020', 5: '01-27-2020', 6: '02-24-2020', 7: '02-09-2020',
                           8: '02-18-2020', 9: '02-14-2020', 10: '02-12-2020'}

    dict_of_eu_geog = {0: 'Austria', 1: 'Belgium', 2: 'Denmark', 3: 'France',
                       4: 'Germany', 5: 'Italy', 6: 'Norway', 7: 'Spain',
                       8: 'Sweden', 9: 'Switzerland', 10: 'United_Kingdom'}

    path = "../results/europe_summary.csv"

    for plot_choice in range(0, 2):
        for num_of_country in dict_of_eu_geog.keys():
            confirmed_cases = read_true_cases_europe(plot_choice, num_of_country, dict_of_start_dates, dict_of_eu_geog)
            plot_daily_infections_num(path, confirmed_cases, "", plot_choice, num_of_country, dict_of_start_dates,
                                      dict_of_eu_geog)
    return


def main():
    if plot_settings == 'usa':
        make_all_us_plots()
    if plot_settings == 'eu':
        make_all_eu_plots()


if __name__ == '__main__':
    main()
