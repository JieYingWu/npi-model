from os.path import join
import sys
import torch
import torch.nn as nn
from loss import PoissonLoss
import numpy as np

FEATURES = 9
HIDDEN_DIM = 16
OUTPUT_DIM = 1

# Load some data


lr = 1e-4
batch = 16


model = NpiLstm(FEATURES, HIDDEN_DIM, batch, OUTPUT_DIM)
loss_fn = torch.nn.PoissonLoss()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
 
for t in range(num_epochs):
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
    optimiser.zero_grad()
 
    # Backward pass
    loss.backward()
 
    # Update parameters
    optimiser.step()
