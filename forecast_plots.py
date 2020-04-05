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


def plot_forecast_by_cols(Xconf, Yconf, Xpred, Ypred):
    '''
    :param Xconf: datestamps of confirmed cases
    :param Yconf: values of confirmed cases
    :param Xpred: datestamps of predicted cases
    :param Ypred: values of predicted cases
    '''

    # plot prediction with 50 of assurance intervall
    y1_upper = np.asarray(Ypred) * 1.25
    y1_lower = np.asarray(Ypred) * 0.75
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(Xpred, Ypred, '-g', alpha=0.6)  # solid green
    ax.plot(Xpred, y1_lower, '-c', alpha=0.2)
    ax.plot(Xpred, y1_upper, '-c', alpha=0.2)
    ax.fill_between(Xconf, y1_lower, y1_upper, alpha=0.2)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # plot confirmed as a barplot
    ax.bar(Xconf, Yconf, color='g', width=0.3, alpha=0.3)
    ax.set_ylabel("Deaths")
    ax.set_xlabel("Date")
    plt.show()


def plot_forecasts(data_country):
    '''
    :param data_country: pandas DF that contains column 'deaths' and 'time'
    '''
    df = data_country

    y1_upper = np.asarray(df['deaths'] * 1.25)
    y1_lower = np.asarray(df['deaths'] * 0.75)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(data_country['time'],data_country['deaths'],'-g', alpha=0.6)  # solid green
    ax.plot(data_country['time'],y1_lower,'-c', alpha=0.2)
    ax.plot(data_country['time'],y1_upper,'-c', alpha=0.2)
    ax.fill_between(data_country['time'], y1_lower, y1_upper, alpha=0.2)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax.bar(data_country['time'],data_country['deaths'],color='g',width=0.8,alpha=0.3)
    ax.set_ylabel("Deaths")
    ax.set_xlabel("Date")

    save_location = './results/plots/uk.jpg'
    plt.savefig(fname = save_location)

def plot_forecasts_without_dates(row):
    '''
    :param data_country: pandas DF that contains column 'deaths' and 'time'
    '''
    ticks = range(0, np.shape(row)[0])
    y1_upper = np.asarray(row * 1.25)
    y1_lower = np.asarray(row * 0.75)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(ticks, row, '-g', alpha=0.6)  # solid green
    ax.plot(ticks, y1_lower, '-c', alpha=0.2)
    ax.plot(ticks, y1_upper, '-c', alpha=0.2)
    ax.fill_between(ticks, y1_lower, y1_upper, alpha=0.2)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax.bar(ticks, row, color='g', width=0.8, alpha=0.3)
    ax.set_ylabel("Deaths")
    ax.set_xlabel("Date")

    plt.show()


def trial_run():
    # fill with dumb data
    dates = ['2020-03-16', '2020-03-17', '2020-03-18',
         '2020-03-19', '2020-03-20', '2020-03-21',
         '2020-03-22', '2020-03-23', '2020-03-24',
         '2020-03-25', '2020-03-26', '2020-03-27']

    deaths_predicted = [1, 5, 10, 20, 100, 200, 250, 380, 500, 510, 520, 550]
    deaths_confirmed = [1, 5, 10, 20, 100, 190, 220]
    data_country = {'time':	dates,
                'deaths': deaths_predicted}

    df = pd.DataFrame(data=data_country)
    # example usage
    plot_forecasts(df)

df = pd.read_csv(r"D:\JHU\corona\npi-model\npi-model\summary_europe.csv", delimiter=';',index_col=0)
#print(df.head())
row_names = list(df.index.tolist())
#print(row_names)
prediction_list = []
county_number = '1]'
for name in row_names:
    if "prediction[" in name:
        if name.split(",")[1] == county_number:
            print(name)
            rowData = df.loc[name, :]
            prediction_list.append(rowData['mean'])
prediction_list = np.array(prediction_list)
plot_forecasts_without_dates(prediction_list)