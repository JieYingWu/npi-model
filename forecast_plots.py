from os.path import join, exists
import sys
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def plot_forecasts(data_country):
    y1_upper = np.asarray(df['deaths'] * 1.25)
    y1_lower = np.asarray(df['deaths'] * 0.75)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(data_country['time'],data_country['deaths'],'-g', alpha=0.6)  # solid green
    ax.plot(data_country['time'],y1_lower,'-c', alpha=0.2)
    ax.plot(data_country['time'],y1_upper,'-c', alpha=0.2)
    ax.fill_between(data_country['time'], y1_lower, y1_upper, alpha=0.2)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax.bar(data_country['time'],data_country['deaths'],color='g',width=0.3,alpha=0.3)
    ax.set_ylabel("Deaths")
    ax.set_xlabel("Date")

    plt.show()


# fill with dumb data
dates = ['2020-03-16', '2020-03-17', '2020-03-18',
         '2020-03-19', '2020-03-20', '2020-03-21',
         '2020-03-22', '2020-03-23', '2020-03-24']

deaths = [1, 5, 10, 20, 100, 200, 250, 380, 500]
data_country = {'time':	dates,
                'deaths': deaths}

df = pd.DataFrame(data=data_country)
plot_forecasts(df)