from os.path import join, exists
import sys
import numpy as np

data_dir = sys.argv[1]

countries = ['Denmark', 'Italy', 'Germany', 'Spain', 'United Kingdom', 'France', 'Norway', 'Belgium', 'Austria', 'Sweden', 'Switzerland']
N = 75
serial.interval = read.csv(join(data_dir, 'serial_interval.csv')) # Time between primary infector showing symptoms and secondary infectee showing symptoms
interventions = np.loadtxt(join(data_dir, 'interventions.csv'))

                           
