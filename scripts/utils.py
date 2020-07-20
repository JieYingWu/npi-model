import numpy as np
import pandas as pd
from os.path import join

def get_cluster(filename, cluster_num):
    df = pd.read_csv(filename)
    fips = df.loc[df['cluster'] == cluster_num, 'FIPS']
    return fips.values

def calculate_rt(r0, alphas, interventions):
    rt = r0 - alphas * covariates
    return rt

def calculate_cases(rt, population, si):
    n = len(rt)
    cases = np.zeros(n) 
    for i in range(2,n):
        case = np.sum(infected[0:i] * si[i-1::-1])
        rt_adj = (population -np.sum(infected)) / population * rt[i]
        cases[i] = rt_adj * case
    return cases

def calculate_deaths(cases, ifr, fatality):
    n = len(rt)
    deaths = np.zeros(n)
    deaths[0] = 1e-15*infected[0]
    for i in range(1,n):
        death = np.sum(cases[0:i] * fatality[i-1::-1])
        deaths[i] = ifr * death
    return deaths    

def get_npis(data_dir):
    interventions_path = join(data_dir, 'us_data/interventions.csv')
    interventions = pd.read_csv(interventions_path)
    id_cols = ['FIPS', 'STATE', 'AREA_NAME']    
    int_cols = [col for col in interventions.columns.tolist() if col not in id_cols]
    interventions.drop(columns=['STATE', 'AREA_NAME'], inplace=True)

    interventions.drop([0], axis=0, inplace=True)
    interventions.fillna(1, inplace=True)
    return interventions

def get_counties_isolated_NPIs(npis, index):
    cur_npi_dates = np.array(npis[index].values)
    dates_diff =  abs(np.array(npis.iloc[:,1:].values).transpose() - cur_npi_dates).transpose()
    dates_diff.sort(axis=1)
    to_keep = npis[dates_diff[:,1] > 2]
    return to_keep['FIPS']


def compute_moving_window(x, window_size, axis=0, mode='left', func='mean'):
    """Compute the moving average

    :param x: 
    :param window_size: 
    :param axis: axis to take the window over.
    :param mode: one of 'left', 'center', or 'right'. Which side of the current entry to take the average over.
    :param func: one of 'mean', 'std'
    :rtype: 

    """
    assert mode in ['left', 'center', 'right']
    x = np.array(x)

    if mode == 'center':
        assert window_size % 2 == 1, 'use an odd-numbered window size for mode == \'center\''
    window_width = window_size // 2
    
    y = np.empty_like(x)
    for i in range(x.shape[axis]):
        if mode == 'left':
            seq = tuple(np.arange(max(i - window_size + 1, 0), i+1) if ax == axis else slice(x.shape[ax]) for ax in range(x.ndim))
        elif mode == 'center':
            seq = tuple(np.arange(max(i - window_width + 1, 0), min(i + window_width + 1, x.shape[ax])) if ax == axis else slice(x.shape[ax]) for ax in range(x.ndim))
        elif mode == 'right':
            seq = tuple(np.arange(i, min(i + window_size, x.shape[ax])) if ax == axis else slice(x.shape[ax]) for ax in range(x.ndim))
        else:
            raise ValueError

        yidx = tuple(i if ax == axis else slice(y.shape[ax]) for ax in range(y.ndim))
        # print('yidx:', yidx)
        if func == 'mean':
            # print('seq:', seq)
            # print('x[seq]:', x[seq])
            y[yidx] = x[seq].mean(axis)
            # print('y[yidx]:', y[yidx])
            # if (y[yidx] > 0).mean() > 0.5:
            #     raise ValueError
        elif func == 'std':
            y[yidx] = x[seq].std(axis)
    return y


def compute_moving_average(*args, **kwargs):
    kwargs['func'] = 'mean'
    return compute_moving_window(*args, **kwargs)


def compute_moving_std(*args, **kwargs):
    kwargs['func'] = 'std'
    return compute_moving_window(*args, **kwargs)

    
if __name__ == '__main__':
#    filename = 'data/us_data/clustering.csv'
#    cluster_num = 5
#    fips = get_cluster(filename, cluster_num)
#    print(fips)

    
    data_dir = 'data'
    interventions = get_npis(data_dir)
    counties = get_counties_isolated_NPIs(interventions, 'public schools')
    print(counties)
