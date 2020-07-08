import os
from os.path import join, exists
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_rhat(path, filename):
    df = pd.DataFrame()
    df = pd.read_csv(join(path, 'summary.csv'), delimiter=',', index_col=0)
    param = ['mu', 'alpha']

    param_rhat = []
    derived_rhat = []

    row_names = list(df.index.tolist())
    for name in row_names:
        if 'mu' in name or 'alpha' in name:
            param_rhat.append(df.loc[name, 'Rhat'])
        else:
            # Sometimes deaths are not sampled but seeded so gives nan
            if ~np.isnan(df.loc[name, 'Rhat']):
                derived_rhat.append(df.loc[name, 'Rhat'])
                
    param_rhat = np.array(param_rhat)
    param_hist, param_bin_edges = np.histogram(param_rhat)
    derived_rhat = np.array(derived_rhat)
    derived_hist, derived_bin_edges = np.histogram(derived_rhat)
    
    fig = plt.figure('Parameter Estimates Rhat')
    plt.bar(param_bin_edges[:-1], param_hist, width = 0.5, color='#0504aa',alpha=0.7)
    plt.xlim(min(param_bin_edges), max(param_bin_edges))
    plt.xlabel('Rhat',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()
    fig.savefig(join(path, filename+'_param_rhat.pdf'))
    fig.clf()

    fig = plt.figure('Derived Estimates Rhat')
    plt.bar(derived_bin_edges[:-1], derived_hist, width = 0.5, color='#0504aa',alpha=0.7)
    plt.xlim(min(derived_bin_edges), max(derived_bin_edges))
    plt.xlabel('Rhat',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()
    fig.savefig(join(path, filename+'_derived_rhat.pdf'))
    fig.clf()
    
def main(path, name):
    read_rhat(path, name)
    
if __name__ == '__main__':
    # run from base directory 
    main(sys.argv[1], sys.argv[2])
