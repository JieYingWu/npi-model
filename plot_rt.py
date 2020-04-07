import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# otherwise rotation of x-labels by 90 deg crashes from time to time
ticker.Locator.MAXTICKS = 10000


def plot_rt(simulation_file, interventions_file, country_number, num_days=75, start_date='2020-02-22', save_img=False):
    # read data
    simulation_data = pd.read_csv(simulation_file, delimiter=';', index_col=0)
    countries, interventions, interventions_data = get_interventions(interventions_file)
    interventions_data = interventions_data[interventions_data['Country'] == countries[country_number - 1]]
    time_data = list(pd.date_range(start=start_date, periods=num_days))
    # remove those interventions, that are not considered in their report
    interventions.remove('sport')
    interventions.remove('travel_restrictions')
    interventions.remove('any government intervention')

    start = 'Rt[1,' + str(country_number) + ']'
    end = 'Rt[' + str(num_days) + ',' + str(country_number) + ']'
    Rt_data = simulation_data.loc[start:end]

    plt.figure()
    # plot mean
    #plt.plot(time_data, Rt_data['mean'], drawstyle='steps-post', color='darkgreen')
    # horizontal line at R=1
    plt.hlines(1, time_data[0], time_data[-1])

    # 50% conf interval
    upper = Rt_data['75%']
    lower = Rt_data['25%']
    plt.fill_between(time_data, lower, upper, step='post', alpha=0.4, color='darkgreen', label='50% conf. interval')

    # 95% conf interval
    upper = Rt_data['97.5%']
    lower = Rt_data['2.5%']
    plt.fill_between(time_data, lower, upper, step='post', alpha=0.4, color='lightgreen', label='95% conf. interval')

    init_height = 1.2 * plt.gca().get_ylim()[1]
    plt.ylim([0, init_height])

    # markers used for different kinds of interventions
    marker = ['o', 'v', '^', '<', '>', 's', '*', 'D']
    # neede to make sure marker for several interventions on the same day don't overlap
    adjust_height = []

    # plot vertical lines and markers for each intervention
    # make sure markers for several interventions on the same day are drawn at different positions
    for ind, intervention in enumerate(interventions):
        date = interventions_data[intervention].values[0]
        num = adjust_height.count(date)
        adjust_height.append(date)
        if num == 0:
            # plot vertical line
            plt.axvline(pd.to_datetime(date), drawstyle='steps-pre', color='gray', ls='--')
        # plot amrker for this intervention
        plt.plot(pd.to_datetime(date), init_height - (num + 1) * 0.05 * init_height, marker=marker[ind],
                 label=intervention, linestyle='None')

    # legend for interventions
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # adjust x axis labels and figure size
    date_form = DateFormatter("%B %d")
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.30)
    plt.xlim([time_data[0], time_data[-1]])

    plt.title(countries[country_number - 1])
    plt.xlabel('Days')
    plt.ylabel('Time-dependent Reproduction Number')

    if save_img:
        plt.savefig('./europe_results/plots/Rt_{}.png'.format(countries[country_number - 1]), bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.show()


# copied from data_parser
def get_interventions(interventions_file):

    interventions = pd.read_csv(interventions_file, encoding='latin-1')

    mod_interventions = pd.DataFrame(columns=['Country', 'school/uni closures', 'self-isolating if ill',
                                              'banning public events', 'any government intervention',
                                              'complete/partial lockdown', 'social distancing/isolation'])

    mod_interventions['Country'] = interventions.iloc[0:11]['Country']
    mod_interventions['school/uni closures'] = interventions.iloc[0:11]['schools_universities']
    mod_interventions['self-isolating if ill'] = interventions.iloc[0:11]['self_isolating_if_ill']
    mod_interventions['banning public events'] = interventions.iloc[0:11]['public_events']
    mod_interventions['social distancing/isolation'] = interventions.iloc[0:11]['social_distancing_encouraged']
    mod_interventions['complete/partial lockdown'] = interventions.iloc[0:11]['lockdown']
    mod_interventions['any government intervention'] = interventions.iloc[0:11]['lockdown']
    mod_interventions['sport'] = interventions.iloc[0:11]['sport']
    mod_interventions['travel_restrictions'] = interventions.iloc[0:11]['travel_restrictions']

    mod_interventions.sort_values('Country', inplace=True)

    for col in mod_interventions.columns.tolist():
        if col == 'Country' or col == 'complete/partial lockdown':
            continue
        col1 = pd.to_datetime(mod_interventions[col], format='%Y-%m-%d').dt.date
        col2 = pd.to_datetime(mod_interventions['complete/partial lockdown'], format='%Y-%m-%d').dt.date
        col3 = pd.to_datetime(mod_interventions['any government intervention'], format='%Y-%m-%d').dt.date
        mod_interventions[col] = np.where(col1 > col2, col2, col1).astype(str)
        if col != 'self-isolating if ill':
            mod_interventions['any government intervention'] = np.where(col1 < col3, col1, col1).astype(str)

    countries = mod_interventions['Country'].to_list()
    date_cols = [col for col in mod_interventions.columns.tolist() if col != 'Country']

    return countries, date_cols, mod_interventions


if __name__ == '__main__':

    simulation_file = r'summary_europe.csv'
    interventions_file = r'data\interventions.csv'
    country_numbers = np.arange(1, 12)
    start_dates = ['02-22-2020', '02-18-2020', '02-21-2020', '02-07-2020', '02-15-2020', '01-27-2020', '02-24-2020', '02-09-2020', '02-18-2020', '02-14-2020', '02-12-2020']

    for country, date in zip(country_numbers, start_dates):
        plot_rt(simulation_file, interventions_file, country, start_date=date, save_img=True)



