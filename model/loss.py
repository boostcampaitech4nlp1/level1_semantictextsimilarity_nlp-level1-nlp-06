import torch.nn as nn

def L1_loss(output, target):
    loss = nn.L1Loss()
    
    return loss(output, target)

def MSE_loss(output, target):
    mse_Loss = nn.MSELoss()
    
    return mse_Loss(output, target)

def Huber_loss(output, target):
    loss = nn.HuberLoss()
    
    return loss(output, target)