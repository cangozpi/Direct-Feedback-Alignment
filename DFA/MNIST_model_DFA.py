import torch
from torch import nn
from torch.autograd import grad

# Model architecture for MNIST dataset
class DFA(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = DFA_Linear(28*28, 100) 
        self.linear2 = DFA_Linear(100, 50) 
        self.final = DFA_Linear(50, 10)
        self.relu = nn.ReLU()
        
        

    def forward(self, image):
        a = image.view(-1, 28*28)
        a = self.relu(self.linear1(a))
        a = self.relu(self.linear2(a))
        a = self.final(a) 
    
        return a
        
    def loss_distributer(self,loss,output):
        loss_gradient = grad(loss, output, retain_graph=True)[0]
        # Broadcast gradient of the loss to every layer
        for layer in self.modules():
            layer.loss_gradient = loss_gradient
    


# DFA Linear Layer
class DFA_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None,dtype=None):
        super(DFA_Linear, self).__init__(in_features,out_features,bias,device,dtype)
        
        # Initialize Backward weight matrices used for backward pass in DFA
        self.backward_weight = nn.Parameter(torch.Tensor(size=(out_features, in_features)), requires_grad=False) # B
        if self.bias is not None:
            self.backward_bias = nn.Parameter(torch.Tensor(size=(out_features, in_features)), requires_grad=False)

        # Globally propagate error signals as in DFA
        self.register_full_backward_hook(self.dfa_backward_hook) 
        

    
    
    
    @staticmethod
    def dfa_backward_hook(module, grad_input, grad_output):
        # If layer don't have grad w.r.t input
        if grad_input[0] is None:
            return grad_input
        else:
            grad_dfa = module.loss_gradient.mm(module.backward_weight)
            # If no bias term
            
            if len(grad_input) == 2:
                return grad_dfa, grad_input[0]
            else:
                return grad_dfa, grad_input[0], grad_input[1]

