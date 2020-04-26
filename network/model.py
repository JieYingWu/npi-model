import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
                
    def forward(self, x):        
        y, (hidden, cell) = self.lstm(x)
        return hidden, cell
    

class NpiLstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch, N2=100, output_dim=1,
                    num_layers=2, device='cpu'):
        super(NpiLstm, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch = batch
        self.num_layers = num_layers
        self.N2 = N2
        self.device = device
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
 
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
 
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.init_hidden()

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.hidden =  (torch.zeros(self.num_layers, self.batch, self.hidden_dim),
                        torch.zeros(self.num_layers, self.batch, self.hidden_dim))
        
        
    def forward(self, x):

        # Use the first N days to get cell state
        hidden, cell = self.encoder(src)
        output = torch.zeros(self.N2, self.batch).to(self.device)

        # Predict to N2 - probably shouldn't start at 1? But how to do county specific
        for t in range(1, self.N2):
            y, (hidden, cell) = self.lstm(x)
            output[t] = y
        
        return output
    
