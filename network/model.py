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
                    num_layers=1, device='cpu'):
        super(NpiLstm, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch = batch
        self.num_layers = num_layers
        self.N2 = N2
        self.device = device
#        self.encoder = Encoder(self.input_dim, self.hidden_dim)
 
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)
 
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.init_hidden()

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.hidden = torch.zeros(self.num_layers, self.batch, self.hidden_dim).to(self.device)
        self.cell = torch.zeros(self.num_layers, self.batch, self.hidden_dim).to(self.device)
        
        
    def forward(self, x):
        # Use the first N days to get cell state
#        self.hidden = self.encoder(x)
        output = torch.zeros(x.size()[0], self.batch).to(self.device) + 1e-9
        
        # Predict to N2 - probably shouldn't start at 1? But how to do county specific
        for t in range(1, x.size()[0]):

            cur_x = x[t].unsqueeze(0)
            
            # Concatenate with last output
            cur_in = torch.cat((cur_x, output[t-1].unsqueeze(0).unsqueeze(2)), axis=2)
            self.hidden, self.cell = self.lstm(cur_in, (self.hidden, self.cell))
            print(self.hidden)
            print(self.cell)
            rt_pred = self.linear(self.hidden)
            output[t] = rt_pred.squeeze()
        
        return output
    
