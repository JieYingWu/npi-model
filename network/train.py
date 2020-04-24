from os.path import join
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss import PoissonLoss
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

FEATURES = 9
HIDDEN_DIM = 16
OUTPUT_DIM = 1

# Make dataloaders
train_dataset = NpiDataset()
train_loader = DataLoader(dataset=train_dataset, batch_size=bach, shuffle=False, num_workers=4)
train_dataset = NpiDataset()
train_loader = DataLoader(dataset=train_dataset, batch_size=bach, shuffle=False, num_workers=4)

# Read weighted fatalities and serial interval
wf_file = join(data_dir, 'us_data', 'weighted_fatality.csv')
weighted_fatalities = np.loadtxt(wf_file, skiprows=1, delimiter=',', dtype=str)
ifrs = {}
for i in range(weighted_fatalities.shape[0]):
    ifrs[weighted_fatalities[i,0]] = weighted_fatalities[i,-1]
serial_interval = np.loadtxt(join(data_dir, 'serial_interval.csv'), skiprows=1, delimiter=',')

lr = 1e-4
batch = 16
n_epochs = 500

model = NpiLstm(FEATURES, HIDDEN_DIM, batch, OUTPUT_DIM)
print(f'The model has {count_parameters(model):,} trainable parameters')

loss_fn = torch.nn.PoissonLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer)

all_loss = np.zeros(num_epochs)

for e in range(n_epochs):
    model.train()
    epoch_loss = 0
    for i, () in enumerate(train_loader):
        
        
        # Forward pass
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        all_loss[t] = loss.item()

        # Backward pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

        # Validate
        if e % validate_each == 0:

            
