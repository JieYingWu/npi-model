from os.path import join, exists
import sys
import numpy as np

import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## Copied from https://towardsdatascience.com/an-introduction-to-bayesian-inference-in-pystan-c27078e58d53
sns.set()  # Nice plot aesthetic
np.random.seed(101)

model = """

data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(alpha + beta * x, sigma);
}

"""

# Parameters to be inferred
alpha = 4.0
beta = 0.5
sigma = 1.0

# Generate and plot data
x = 10 * np.random.rand(100)
y = alpha + beta * x
y = np.random.normal(y, scale=sigma)

# Put our data in a dictionary
data = {'N': len(x), 'x': x, 'y': y}

# Compile the model
sm = pystan.StanModel(model_code=model)

# Train the model and generate samples
fit = sm.sampling(data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101)


# data_dir = sys.argv[1]

# countries = ['Denmark', 'Italy', 'Germany', 'Spain', 'United Kingdom', 'France', 'Norway', 'Belgium', 'Austria', 'Sweden', 'Switzerland']
# N = 75
# serial.interval = read.csv(join(data_dir, 'serial_interval.csv')) # Time between primary infector showing symptoms and secondary infectee showing symptoms
# interventions = np.loadtxt(join(data_dir, 'interventions.csv'))


