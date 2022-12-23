import torch
from torch import nn

# DFA Linear Layer
class DFA_Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int,output_dim: int, bias = True, device = None, dtype = None):
        super(DFA_Linear, self).__init__(in_features,out_features,bias,device,dtype)
        
        # Initialize Backward weight matrices used for backward pass in DFA
        self.backward_weight = nn.Parameter(torch.Tensor(size = (output_dim,in_features)), requires_grad = False) # B --> same shape as W transpose
        if self.bias is not None:
            self.backward_bias = nn.Parameter(torch.Tensor(size = (output_dim,in_features)), requires_grad = False)

        # Globally propagate error signals as in DFA
        self.register_full_backward_hook(self.dfa_backward_hook) 
        

    
    
    
    @staticmethod
    def dfa_backward_hook(module, grad_input, grad_output):
        """
        NOTE: grad_input[0]: grad wrt inputs(X)
        """
        #print(module, type(module))
        #print(len(grad_input), type(grad_input))
        
        # If layer don't have grad w.r.t input (e.g. activation layers)
        if grad_input[0] is None:
            return grad_input
        else:
            # grad_dfa = torch.matmul(module.backward_weight, module.global_loss_gradient.T) # B*e --> [in_features, B]
            grad_dfa = torch.matmul(module.global_loss_gradient, module.backward_weight) # B*e --> [in_features, B]
            #print(f"HOHOHO {type(tuple(grad_dfa.T))} {grad_dfa.T.shape} {len(grad_input)}")
            return (grad_dfa, ) 

