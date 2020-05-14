from os.path import join, exists
import os
import numpy as np
import pandas as pd
import datetime as dtype

interventions_path = join('data', 'us_data', 'interventions.csv')
interventions = pd.read_csv(interventions_path)

interventions = interventions.dropna()

shelter = interventions['stay at home'].values

gather50 = shelter - interventions['>50 gatherings'].values
gather500 = shelter - interventions['>500 gatherings'].values
schools = shelter - interventions['public schools'].values
restaurant = shelter - interventions['restaurant dine-in'].values
entertainment = shelter - interventions['entertainment/gym'].values
federal = shelter - interventions['federal guidelines'].values
travel = shelter - interventions['foreign travel ban'].values



import matplotlib.pyplot as plt

output_path = join('results', 'intervention_dates')
if not exists(output_path):
    os.makedirs(output_path)


plt.hist(gather50, bins=len(set(gather50)))
plt.gca().set(title='>50 gathering', ylabel='Frequency');
plt.tight_layout()
plt.savefig(join(output_path, '50_gathering.png'))
plt.clf()

plt.hist(gather500, bins=len(set(gather500)))
plt.gca().set(title='>500 gathering', ylabel='Frequency');
plt.tight_layout()
plt.savefig(join(output_path, '500_gathering.png'))
plt.clf()

plt.hist(schools, bins=len(set(schools)))
plt.gca().set(title='Public schools', ylabel='Frequency');
plt.tight_layout()
plt.savefig(join(output_path, 'schools.png'))
plt.clf()

plt.hist(restaurant, bins=len(set(restaurant)))
plt.gca().set(title='Restaurant dine-in', ylabel='Frequency');
plt.tight_layout()
plt.savefig(join(output_path, 'restaurant.png'))
plt.clf()

plt.hist(entertainment, bins=len(set(entertainment)))
plt.gca().set(title='Entertainment/gym', ylabel='Frequency');
plt.tight_layout()
plt.savefig(join(output_path, 'entertainment.png'))
plt.clf()

plt.hist(federal, bins=len(set(federal)))
plt.gca().set(title='Federal guidelines', ylabel='Frequency');
plt.tight_layout()
plt.savefig(join(output_path, 'federal.png'))
plt.clf()

plt.hist(travel, bins=len(set(travel)))
plt.gca().set(title='Foreign travel ban', ylabel='Frequency');
plt.tight_layout()
plt.savefig(join(output_path, 'travel.png'))
plt.clf()

