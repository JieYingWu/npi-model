import os
from os.path import join, exists
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def read_rhat(path):
    param_rhat = []
    derived_rhat = []
    all_rhat = []

    for i in range(5):
        df = pd.DataFrame()
        df = pd.read_csv(join(path, 'cluster_' + str(i), 'summary.csv'), delimiter=',', index_col=0)
        param = ['mu', 'alpha', 'mask']

        row_names = list(df.index.tolist())
        for name in row_names:
            if ~np.isnan(df.loc[name, 'Rhat']):
                all_rhat.append(df.loc[name, 'Rhat'])
            if 'mu' in name or 'alpha' in name:
                if ~np.isnan(df.loc[name, 'Rhat']):
                    param_rhat.append(df.loc[name, 'Rhat'])
            else:
                # Sometimes deaths are not sampled but seeded so gives nan
                if ~np.isnan(df.loc[name, 'Rhat']):
                    derived_rhat.append(df.loc[name, 'Rhat'])

    param_rhat = np.array(param_rhat)
    param_hist, param_bin_edges = np.histogram(param_rhat)
    derived_rhat = np.array(derived_rhat)
    derived_hist, derived_bin_edges = np.histogram(derived_rhat)
    all_rhat = np.array(all_rhat)
    all_hist, all_bin_edges = np.histogram(all_rhat)

    fig = plt.figure('Parameter Estimates Rhat')
    sns.distplot(param_rhat, ax=plt.gca(), kde=False, color='#0504aa')
    plt.xlabel('Rhat')
    plt.ylabel('Frequency')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    fig.savefig(join(path, 'param_rhat.pdf'))
    fig.clf()
    
    fig = plt.figure('Derived Estimates Rhat')
    sns.distplot(derived_rhat, ax=plt.gca(), kde=False, color='#0504aa')
    # plt.bar(derived_bin_edges[:-1], derived_hist, width = 0.5, color='#0504aa',alpha=0.7)
    plt.xlabel('Rhat')
    plt.ylabel('Frequency')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    fig.savefig(join(path, 'derived_rhat.pdf'))
    fig.clf()

    fig = plt.figure('Rhat')
    sns.distplot(all_rhat, ax=plt.gca(), kde=False, color='#0504aa')
    # plt.bar(derived_bin_edges[:-1], derived_hist, width = 0.5, color='#0504aa',alpha=0.7)
    plt.xlabel('Rhat')
    plt.ylabel('Frequency')
#    plt.xticks(fontsize=15)
#    plt.yticks(fontsize=15)
    plt.tight_layout()
    fig.savefig(join(path, 'all_rhat.pdf'))
    fig.clf()

    
if __name__ == '__main__':
    # run from base directory 
    read_rhat(sys.argv[1])
