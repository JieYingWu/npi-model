import sys
import torch
import numpy as np
import torch.nn as nn
from os.path import join
from loss import PoissonLoss
from model import NpiLstm
from dataloader import LSTMDataset

FEATURES = 9
HIDDEN_DIM = 16
OUTPUT_DIM = 1
num_epochs = 1

lr = 1e-4
batch = 16

# set up device 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load some data
# Create train and val set

train_set = LSTMDataset(data_dir='data/us_data', split='train', retail_only=True, verbose=True)
val_set = LSTMDataset(data_dir='data/us_data', split='val', retail_only=True, verbose=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch,
                                                    shuffle=False, num_workers=0)
        
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch,
                                                    shuffle=False, num_workers=0)



model = NpiLstm(FEATURES, HIDDEN_DIM, batch, OUTPUT_DIM).to(device)
loss_fn = PoissonLoss
optim = torch.optim.Adam(model.parameters(), lr=lr)
 
for t in range(num_epochs):
    for batch_idx, train_data in enumerate(train_loader):
        model.train()
        
        # X_train = 
        # y_train = train_data['deaths'].to(device)
        # Clear stored gradient
        model.zero_grad()
    
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        model.init_hidden()
        
        # Forward pass
        y_pred = model(X_train)
    
        loss = loss_fn(y_pred, y_train)
        if t % 100 == 0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
    
        # Zero out gradient, else they will accumulate between epochs
        optim.zero_grad()
    
        # Backward pass
        loss.backward()
    
        # Update parameters
        optim.step()
    

