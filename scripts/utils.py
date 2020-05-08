import numpy as np
import pandas as pd

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
        case = np.sum(infected[0:i-1] * si[0:i-1])
        rt_adj = (population -np.sum(infected)) / population * rt[i]
        cases[i] = rt_adj * case
    return cases

def calculate_deaths(cases, ifr, fatality):
    n = len(rt)
    deaths = np.zeros(n)
    deaths[0] = 1e-15*infected[0]
    for i in range(1,n):
        death = np.sum(cases[0:i-1] * fatality[0:i-1])
        deaths[i] = ifr * death
    return deaths    

if __name__ == '__main__':
    filename = 'data/us_data/clustering.csv'
    cluster_num = 5
    fips = get_cluster(filename, cluster_num)
    print(fips)
