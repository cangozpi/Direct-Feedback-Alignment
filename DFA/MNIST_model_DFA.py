import torch
from torch import nn
from torch.autograd import grad
from .layers import *

# Model architecture for MNIST dataset
class DFA(nn.Module):
    def __init__(self, p_drop, use_BatchNorm1D, use_LayerNorm1D):
        """
        Inputs:
            p_drop (float): dropout probability.
            use_BatchNorm1D (bool): If True apply batch norm.
            use_LayerNorm1D (bool): If True apply layer norm.

        """
        super().__init__()
        self.use_BatchNorm1D = use_BatchNorm1D
        self.batchNorm = torch.nn.BatchNorm1d(28*28)

        self.use_LayerNorm1D = use_LayerNorm1D
        self.layerNorm = torch.nn.LayerNorm(28*28)

        self.linear1 = DFA_Linear(28*28, 100, 10) 
        self.dropout1 = torch.nn.Dropout(p_drop)
        self.linear2 = DFA_Linear(100, 50, 10) 
        self.dropout2 = torch.nn.Dropout(p_drop)
        self.final = DFA_Linear(50, 10, 10)
        self.relu = nn.ReLU()
        
        

    def forward(self, image):
        a = image.view(-1, 28*28) # Flatten input --> [B, W*H*C=28*28]

        if self.use_BatchNorm1D:
            a = self.batchNorm(a) # --> [B, W*H*C=28*28]
        
        if self.use_LayerNorm1D:
            a = self.layerNorm(a) # --> [B, W*H*C=28*28]

        a = self.relu(self.dropout1(self.linear1(a)))
        a = self.relu(self.dropout2(self.linear2(a)))
        a = self.final(a) 
    
        return a
        
    def loss_distributer(self, loss, output):
        """
        Propagates global error signal (e) from the top layer towards the intermediate layers
        """
        global_loss_gradient = grad(loss, output, retain_graph=True)[0] # [B, ouput_dim]
        # Broadcast global gradient (e) to every layer
        for layer in self.modules():
            layer.global_loss_gradient = global_loss_gradient
    

