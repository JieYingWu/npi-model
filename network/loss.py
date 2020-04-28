import numpy as np
import torch
import torch.nn as nn
from os.path import exists
from statsmodels.distributions.empirical_distribution import ECDF

class NpiLoss(nn.Module):

    def __init__(self, N2, regions, ifrs, si, device):
        super(NpiLoss, self).__init__()
        self.N2 = N2
        self.ifrs = ifrs
        self.si = torch.from_numpy(si).to(device).type(torch.float32)
        self.regions = regions
        self.M  = len(regions)
        self.device = device

        self.calculate_fatality_rate()
        self.loss_fn = nn.MSELoss()
    
    def poissonLoss(predicted, observed):
        """Custom loss function for Poisson model."""
        
        loss=torch.mean(predicted-observed*torch.log(predicted))
        return loss


    def calculate_fatality_rate(self):
        name = 'cached_loss_' + str(self.N2) + '.csv'
        if exists(name):
            self.f = np.loadtxt(name)
            return
            
        SI = self.si[0:self.N2,1]

        # infection to onset
        mean1 = 5.1
        cv1 = 0.86
        alpha1 = cv1**-2
        beta1 = mean1/alpha1
        # onset to death 
        mean2 = 18.8
        cv2 = 0.45
        alpha2 = cv2**-2
        beta2 = mean2/alpha2
    
        all_f = np.zeros((self.N2, len(self.regions)))
        for r in range(len(self.regions)):
            ifr = float(self.ifrs[str(self.regions[r])])
    
            ## assume that IFR is probability of dying given infection
            x1 = np.random.gamma(alpha1, beta1, 5000000) # infection-to-onset -> do all people who are infected get to onset?
            x2 = np.random.gamma(alpha2, beta2, 5000000) # onset-to-death
            f = ECDF(x1+x2)
            def conv(u): # IFR is the country's probability of death
                return ifr * f(u)

            h = np.zeros(self.N2) # Discrete hazard rate from time t = 1, ..., 100
            h[0] = (conv(1.5) - conv(0.0))

            for i in range(1, self.N2):
                h[i] = (conv(i+.5) - conv(i-.5)) / (1-conv(i-.5))
            s = np.zeros(self.N2)
            s[0] = 1
            for i in range(1, self.N2):
                s[i] = s[i-1]*(1-h[i-1])
                
            all_f[:,r] = s * h
        self.f = all_f.to(device)

        np.savetxt(name, all_f)

    def predict_cases(self, rt):
        prediction = torch.zeros(rt.size()[0], rt.size()[1]).to(self.device)
        for m in range(rt.size()[1]):
            for i in range(rt.size()[0]):
                convolution = 0
                for j in range(i-1):
                    convolution += prediction[j, m] * self.si[i-j]
                prediction[i,m] = rt[i,m] * convolution
        return prediction

    def predict_deaths(self, rt, prediction, idx):
        E_deaths = torch.zeros(rt.size()[0], rt.size()[1]).to(self.device)
        for m in range(len(idx)):
            E_deaths[0,m] = 1e-9
            for i in range(1,rt.size()[0]):
                E_deaths[i,m] = 0;
                for j in range(i-1):
                    E_deaths[i,m] += prediction[j,m] * self.f[i-j,idx[m]]
        return E_deaths

    def forward(self, rt, deaths_gt, idx):
        cases_pred = self.predict_cases(rt)
        deaths_pred = self.predict_deaths(rt, cases_pred, idx)
        loss = self.loss_fn(deaths_pred, deaths_gt)
        return loss
