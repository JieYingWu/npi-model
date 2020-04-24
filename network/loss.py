import torch


def PoissonLoss(predicted, observed):
    """Custom loss function for Poisson model."""
    loss=torch.mean(predicted-observed*torch.log(predicted))
    return loss
