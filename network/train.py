import sys

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from os.path import join
from loss import NpiLoss
from model import NpiLstm
from dataloader import LSTMDataset
import pandas as pd

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

save = lambda ep, model, model_path, error, optimizer, scheduler: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, str(model_path))

data_dir = sys.argv[1]

FEATURES = 9
HIDDEN_DIM = 16
OUTPUT_DIM = 1
num_epochs = 1
N2 = 100

lr = 1e-4
batch = 16
use_previous_model = False
train_counties = ['22071','36061','53033','34031','36059','06037','34003','17031','12086','34017','36103','34027','36119','48201','05119','22103','25017','12011','22033','34039','36087','09009','22095','42077','22017','44007','22105','09001','39101','22055']
val_counties = ['22051','36029','22087','09003','26163']
# set up device 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_set = LSTMDataset(data_dir='data/us_data', counties=train_counties, split='train', retail_only=True, verbose=True)
val_set = LSTMDataset(data_dir='data/us_data', counties=val_counties, split='val', retail_only=True, verbose=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch,
                                                    shuffle=False, num_workers=0)
        
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch,
                                                    shuffle=False, num_workers=0)
train_regions = train_set.counties
val_regions = val_set.counties
regions = train_regions + val_regions

# Read existing weights for both G and D models
if use_previous_model:
    model_path = model_root / 'model_{}.pt'.format(epoch_to_use)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch'] + 1
        step = state['step']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        print('Failed to restore model')
        exit()
else:
    epoch = 1
    step = 0
    
# Read weighted fatalities and serial interval
wf_file = join(data_dir, 'us_data', 'weighted_fatality.csv')
weighted_fatalities = pd.read_csv(wf_file, encoding='latin1', index_col='FIPS')
#ifrs = weighted_fatalities[weighted_fatalities['FIPS'].isin(regions)]
ifrs = {}
print(weighted_fatalities.index)
for r in regions:
    ifrs[r] = weighted_fatalities.loc[int(r), 'fatality_rate']
print(ifrs)

serial_interval = np.loadtxt(join(data_dir, 'serial_interval.csv'), skiprows=1, delimiter=',')

lr = 1e-4
batch = 16
n_epochs = 500

model = NpiLstm(N2, FEATURES, HIDDEN_DIM, batch, OUTPUT_DIM)
print(f'The model has {count_parameters(model):,} trainable parameters')

loss_fn = NpiLoss(N2, regions, ifrs, serial_interval, device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer)

all_loss = np.zeros(num_epochs)

for e in range(n_epochs):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        
        # Forward pass
        rt_pred = model(data['deaths'])
        loss = loss_fn(rt_pred, data['deaths'])
        all_loss[t] = loss.item()

        # Backward pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Validate
        if e % validate_each == 0:
            model.eval()
            all_val_loss = []
            with torch.no_grad():
                for j, data in enumerate(val_loader):

                    # Forward pass
                    rt_pred = model(data['deaths'])
                    loss = loss_fn(rt_pred, data['deaths'])
                    all_val_loss.append(loss.item())

            mean_loss = np.mean(all_val_loss)
            scheduler.step(mean_loss)
            model_path = model_root / "model_{}.pt".format(e)
            save(e, model, model_path, mean_loss, optimizer, scheduler)
