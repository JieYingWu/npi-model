from os.path import join, exists
import sys
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ggplot import *

# in case ggplot complains see solution here https://github.com/yhat/ggpy/issues/662
'''
p <- ggplot(data_country) +
    geom_bar(data = data_country, aes(x = time, y = deaths), 
             fill = "coral4", stat='identity', alpha=0.5) + 
    geom_line(data = data_country, aes(x = time, y = estimated_deaths), 
              col = "deepskyblue4") + 
    geom_line(data = data_country_forecast, 
              aes(x = time, y = estimated_deaths_forecast), 
              col = "black", alpha = 0.5) + 
    geom_ribbon(data = data_country, aes(x = time, 
                                         ymin = death_min, 
                                         ymax = death_max),
                fill="deepskyblue4", alpha=0.3) +
    geom_ribbon(data = data_country_forecast, 
                aes(x = time, 
                    ymin = death_min_forecast, 
                    ymax = death_max_forecast),
                fill = "black", alpha=0.35) +
    geom_vline(xintercept = data_deaths$time[length(data_deaths$time)], 
               col = "black", linetype = "dashed", alpha = 0.5) + 
    #scale_fill_manual(name = "", 
    #                 labels = c("Confirmed deaths", "Predicted deaths"),
    #                 values = c("coral4", "deepskyblue4")) + 
    xlab("Date") +
    ylab("Daily number of deaths\n") + 
    scale_x_date(date_breaks = "weeks", labels = date_format("%e %b")) + 
    scale_y_continuous(trans='log10', labels=comma) + 
    coord_cartesian(ylim = c(1, 100000), expand = FALSE) + 
    theme_pubr() + 
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
    guides(fill=guide_legend(ncol=1, reverse = TRUE)) + 
    annotate(geom="text", x=data_country$time[length(data_country$time)]+8, 
             y=10000, label="Forecast",
             color="black")
  print(p)
'''


def plot_forecast(data_country):#, data_country_forecast):
    g_bar = ggplot(data_country, aes(x='time', weight='deaths')) + geom_bar(colour='red', fill='red') + ggtitle("Deaths") #+ stat_smooth(colour='blue', span=0.2)
    #g_line1 = ggplot(aes(x='time', y='estimated_deaths'), data=data_country) + geom_line() #+ stat_smooth(colour='blue', span=0.2)
    #g_line2 = ggplot(aes(x='time', y='estimated_deaths_forecast'), data=data_country_forecast) + geom_line()

    print(g_bar)
    #print(g_line1)
    #print(g_line2)


#here i make somy dummy data
dates = [pd.to_datetime('2020-03-16'), pd.to_datetime('2020-03-17'), pd.to_datetime('2020-03-18'),
         pd.to_datetime('2020-03-19'), pd.to_datetime('2020-03-20'), pd.to_datetime('2020-03-21'),
         pd.to_datetime('2020-03-22'), pd.to_datetime('2020-03-23'), pd.to_datetime('2020-03-24')]

data_country = {'time':	dates,
                'deaths': [1, 5, 10, 20, 100, 160, 170, 210, 300],
                'estimated_deaths':[1, 5, 10, 100, 150, 160, 170, 190, 200]}

df = pd.DataFrame(data=data_country)
plot_forecast(df)#,data_country_forecast)