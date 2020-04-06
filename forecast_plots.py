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
import getpass

dict_of_start_dates = {1: '02/24/2020', 2: '03/17/2020', 3: '02/24/2020'}
num_of_country = 1
days_to_predict = 40


def plot_forecasts_wo_dates_quantiles(row2_5,row25,row50,row75,row97_5, plot_choice, save_image = False):
    '''
    :param data_country: pandas DF that contains column 'deaths' and 'time'
    '''
    if plot_choice == 0:
        metric = "infections"
    elif plot_choice == 1:
        metric = "deaths"

    base = datetime.datetime.strptime(dict_of_start_dates[num_of_country], '%m/%d/%Y')
    date_list = [base + datetime.timedelta(days=x) for x in range(days_to_predict)]

    print(date_list)

    #ticks = range(0, np.shape(row50)[0])
    ticks = date_list
    y1_upper50 = np.asarray(row75)
    y1_lower50 = np.asarray(row25)
    y1_upper25 = np.asarray(row97_5)
    y1_lower25 = np.asarray(row2_5)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(ticks, row50, '-b', alpha=0.5)
    ax.fill_between(ticks, y1_lower25, y1_upper25, alpha=0.25,color='b')
    ax.fill_between(ticks, y1_lower50, y1_upper50, alpha=0.2, color='b')

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.bar(ticks, row50, color='r', width=0.9, alpha=0.3)
    ax.set_ylabel("Daily number of {}".format(metric))
    ax.set_xlabel("Date")

    ax.xaxis_date()
    fig.autofmt_xdate()
    if save_image:
        plt.savefig('./results/plots/{}.jpg'.format(metric))
    plt.show()


def plot_daily_infections_num(path, num_of_country, days_to_predict, plot_choice, base_model):
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

    df = pd.read_csv(path, delimiter=';', index_col=0)
    row_names = list(df.index.tolist())
    prediction_list = []
    list2_5, list25, list50, list75, list97_5 = [], [], [], [], []
    county_number = str(num_of_country)+']'
    print(county_number)

    for name in row_names:
        if plot_name in name:
            if name.split(",")[1] == county_number:
                rowData = df.loc[name, :]
                prediction_list.append(rowData['mean'])
                list2_5.append(rowData['2.5%'])
                list25.append(rowData['25%'])
                list50.append(rowData['50%'])
                list75.append(rowData['75%'])
                list97_5.append(rowData['97.5%'])

                if name.split(",")[0] == (plot_name + str(days_to_predict)):
                    break
    plot_forecasts_wo_dates_quantiles(list2_5, list25, list50, list75, list97_5, plot_choice)


def main():
    path = r"D:\JHU\corona\npi-model\npi-model\summary_europe.csv"
    plot_choice = 1  # 1 for deaths forecast; 0 for infections forecast
    base_model = False  # True for prediction/E_deaths, False for prediction0/E_deaths0
    plot_daily_infections_num(path, num_of_country, days_to_predict, plot_choice, base_model)


if __name__ == '__main__':
    main()

