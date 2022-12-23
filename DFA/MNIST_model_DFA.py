import torch
from torch import nn
from torch.autograd import grad
from .layers import *

# Model architecture for MNIST dataset
class DFA(nn.Module):
    def __init__(self, p_drop):
        """
        Inputs:
            p_drop (float): dropout probability
        """
        super().__init__()
        self.linear1 = DFA_Linear(28*28, 100,10) 
        self.dropout1 = torch.nn.Dropout(p_drop)
        self.linear2 = DFA_Linear(100, 50,10) 
        self.dropout2 = torch.nn.Dropout(p_drop)
        self.final = DFA_Linear(50, 10,10)
        self.relu = nn.ReLU()
        
        

    def forward(self, image):
        a = image.view(-1, 28*28)
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
    

