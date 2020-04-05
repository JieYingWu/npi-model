from os.path import join, exists
import sys
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


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

    plt.show()


def plot_forecasts_wo_dates_quantiles(row2_5,row25,row50,row75,row97_5):
    '''
    :param data_country: pandas DF that contains column 'deaths' and 'time'
    '''
    ticks = range(0, np.shape(row50)[0])
    y1_upper50 = np.asarray(row75)
    y1_lower50 = np.asarray(row25)
    y1_upper25 = np.asarray(row97_5)
    y1_lower25 = np.asarray(row2_5)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(ticks, row50, '-b', alpha=0.5)

    #ax.plot(ticks, y1_lower25, '-g', alpha=0.1)
    #ax.plot(ticks, y1_upper25, '-g', alpha=0.1)
    ax.fill_between(ticks, y1_lower25, y1_upper25, alpha=0.25,color='b')

    #ax.plot(ticks, y1_lower50, '-g', alpha=0.1)
    #ax.plot(ticks, y1_upper50, '-g', alpha=0.1)
    ax.fill_between(ticks, y1_lower50, y1_upper50, alpha=0.2, color='b')

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.bar(ticks, row50, color='r', width=0.9, alpha=0.3)
    ax.set_ylabel("Daily number of infections")
    ax.set_xlabel("Date")

    plt.show()


def plot_daily_infections_num(path, num_of_country, days_to_predict):
    df = pd.read_csv(path, delimiter=';',index_col=0)
    row_names = list(df.index.tolist())
    prediction_list = []
    list2_5, list25, list50, list75, list97_5 = [],[],[],[],[]
    county_number = str(num_of_country)+']'
    for name in row_names:
        if "prediction[" in name:
        #if "E_deaths[" in name:
            if name.split(",")[1] == county_number:
                print(name)
                rowData = df.loc[name, :]
                prediction_list.append(rowData['mean'])
                list2_5.append(rowData['2.5%'])
                list25.append(rowData['25%'])
                list50.append(rowData['50%'])
                list75.append(rowData['75%'])
                list97_5.append(rowData['97.5%'])

                if name.split(",")[0] == ("prediction[" + str(days_to_predict)):
                    break
    plot_forecasts_wo_dates_quantiles(list2_5, list25, list50, list75, list97_5)


path = r"D:\JHU\corona\npi-model\npi-model\summary_europe.csv"
num_of_country = 1
days_to_predict = 35
plot_daily_infections_num(path,num_of_country,days_to_predict)