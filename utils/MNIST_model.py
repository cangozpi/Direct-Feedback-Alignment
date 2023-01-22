import torch
from torch import nn

# Model architecture for MNIST dataset
class DNN(nn.Module):
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

        self.linear1 = nn.Linear(28*28, 100) 
        self.dropout1 = torch.nn.Dropout(p_drop)
        self.linear2 = nn.Linear(100, 50) 
        self.dropout2 = torch.nn.Dropout(p_drop)
        self.final = nn.Linear(50, 10)
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