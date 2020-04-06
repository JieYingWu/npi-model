from os.path import join, exists
import sys
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import datetime

dict_of_start_dates = {0: '02-22-2020', 1: '02-18-2020', 2: '02-21-2020', 3: '02-07-2020',
                       4: '02-15-2020', 5: '01-27-2020', 6: '02-24-2020', 7: '02-09-2020',
                       8: '02-18-2020', 9: '02-14-2020', 10: '02-12-2020'}

dict_of_eu_geog = {0: 'Austria', 1: 'Belgium', 2: 'Denmark', 3: 'France',
                   4: 'Germany', 5: 'Italy', 6: 'Norway', 7: 'Spain',
                   8: 'Sweden', 9: 'Switzerland', 10: 'United_Kingdom'}

# input here variables for different plotting options
num_of_country = 5
plot_choice = 0  # 1 for deaths forecast; 0 for infections forecast
base_model = True  # True for prediction/E_deaths, False for prediction0/E_deaths0
last_day_to_plot = '03-31-2020'  # predict to this date
start_day_of_confirmed = '12-31-2019'  # this will be different for US and Europe


def plot_forecasts_wo_dates_quantiles(row2_5, row25, row50, row75, row97_5, confirmed_cases, save_image=False):
    '''
    :param row2_5: confidence interval 2.5%
    :param row25: confidence interval 25%
    :param row50: confidence interval 50%
    :param row75: confidence interval 75%
    :param row97_5: confidence interval 97.5%
    :param confirmed_cases: real confirmed cases
    :param save_image: True for save, False for not save
    :return: beautiful magestic plot
    '''
    if plot_choice == 0:
        metric = "infections"
    elif plot_choice == 1:
        metric = "deaths"

    base = datetime.datetime.strptime(dict_of_start_dates[num_of_country-1], '%m-%d-%Y')
    days_to_predict = (datetime.datetime.strptime(last_day_to_plot, '%m-%d-%Y') - base).days
    print("Will make plo for {} of days", days_to_predict)
    date_list = [base + datetime.timedelta(days=x) for x in range(days_to_predict)]

    barplot_missing_values = np.zeros(days_to_predict - np.shape(confirmed_cases)[0])
    barplot_values = list(confirmed_cases)+list(barplot_missing_values)

    ticks = date_list
    y1_upper50 = np.asarray(row75)
    y1_lower50 = np.asarray(row25)
    y1_upper25 = np.asarray(row97_5)
    y1_lower25 = np.asarray(row2_5)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ax.plot(ticks, row50, '-b', alpha=0.5)
    ax.fill_between(ticks, y1_lower25, y1_upper25, alpha=0.25,color='b')
    ax.fill_between(ticks, y1_lower50, y1_upper50, alpha=0.2, color='b')

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    # insert here confirmed cases confirmed_cases
    ax.bar(ticks, barplot_values, color='r', width=0.9, alpha=0.3)
    ax.set_ylabel("Daily number of {}".format(metric))
    ax.set_xlabel("Date")
    ax.title.set_text(str(dict_of_eu_geog[num_of_country-1]))

    ax.xaxis_date()
    fig.autofmt_xdate()
    if save_image:
        plt.savefig('./results/plots/{}.jpg'.format(metric))
    plt.show()


def plot_daily_infections_num(path, confirmed_cases):
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

    base = datetime.datetime.strptime(dict_of_start_dates[num_of_country-1], '%m-%d-%Y')
    days_to_predict = (datetime.datetime.strptime(last_day_to_plot, '%m-%d-%Y') - base).days

    df = pd.read_csv(path, delimiter=';', index_col=0)
    row_names = list(df.index.tolist())
    list2_5, list25, list50, list75, list97_5 = [], [], [], [], []
    county_number = str(num_of_country)+']'

    for name in row_names:
        if plot_name in name:
            if name.split(",")[1] == county_number:
                rowData = df.loc[name, :]
                list2_5.append(rowData['2.5%'])
                list25.append(rowData['25%'])
                list50.append(rowData['50%'])
                list75.append(rowData['75%'])
                list97_5.append(rowData['97.5%'])

                if name.split(",")[0] == (plot_name + str(days_to_predict)):
                    break
    plot_forecasts_wo_dates_quantiles(list2_5, list25, list50, list75, list97_5, confirmed_cases)


def read_true_cases_europe():
     # 1 for deaths forecast; 0 for infections forecast
    if plot_choice == 0:
        filepath = r"D:\JHU\corona\npi-model\npi-model\data\COVID-19-up-to-date-cases-clean.csv"
    else:
        filepath = r"D:\JHU\corona\npi-model\npi-model\data\COVID-19-up-to-date-deaths-clean.csv"

    df = pd.read_csv(filepath, delimiter=',',header=None)

    confirmed_start_date = datetime.datetime.strptime(start_day_of_confirmed, '%m-%d-%Y')
    forecast_start_date = datetime.datetime.strptime(dict_of_start_dates[num_of_country-1], '%m-%d-%Y')
    diff = (forecast_start_date - confirmed_start_date).days + 1

    confirmed_cases = df.iloc[num_of_country-1, diff:]
    return confirmed_cases



def main():
    path = r"D:\JHU\corona\npi-model\npi-model\summary_europe.csv"
    #path = r"D:\JHU\corona\npi-model\npi-model\US_summary.csv"
    confirmed_cases = read_true_cases_europe()
    plot_daily_infections_num(path, confirmed_cases)


if __name__ == '__main__':
    main()

