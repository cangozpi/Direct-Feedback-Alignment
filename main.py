# https://drive.google.com/drive/u/1/folders/1-ZT3n_axengnhZjYK5AsY8s3oDK1WFnL
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms 

from utils.custom_dataloaders import load_MNIST
from utils.MNIST_model import DNN
from DFA.MNIST_model_DFA import DFA
from utils.train_test_utils import train_loop, test_loop, set_seed
import time

# HYPERPARAMETERS -------------------- 
batch_size = 128
epochs = 3
lr = 5e-2
verbose = True
backward_method = "DFA" # possible options "DFA" if not Backprop is used
# Regularization Methods ------------------------------------
available_regularizaiton_methods = [
    'Dropout',
    'BatchNorm1D',
    'LayerNorm',
    'Weight Decay',
    'L1 Regularization',
    'Learning Rate Scheduling',
    'Early Stoppping'
]
p_drop = 0.0 # Dropout probability. If set to 0 than no dropout will be applied
use_BatchNorm1D = False # If True apply batch norm
use_LayerNorm1D = False # If True apply layer norm
l1_regularization_lambda = 0 # L1 regularization lambda value. If set to 0 then no L1 regularization applied
l2_regularization_lambda = 0 # L1 regularization lambda value. If set to 0 then no L1 regularization applied
weight_decay = 0 # Weight Decay. If set to 0 than it is regular SGD with no weight decay (set to something btw [1e-1, 1e-7])
lr_schedular = {
    'use_lr_schedular': False, # If True, torch.optim.lr_schedular.ConstantLR is used
    'factor': 0.5, # The number we multiply learning rate until the milestone
    'total_iters': 5 # The number of steps that the scheduler decays the learning rate
}
# ------------------------------------


set_seed(42) # Set seed for reproducibility reasons

# Load MNIST dataset
train_dataloader, test_dataloader, preprocessing_transform = load_MNIST(batch_size)
if backward_method == "DFA":
    model = DFA(p_drop, use_BatchNorm1D, use_LayerNorm1D)
else:
    model = DNN()
optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay = weight_decay)
lr_sched = torch.optim.lr_scheduler.ConstantLR(optimizer, factor = lr_schedular['factor'], total_iters = lr_schedular['total_iters']) if lr_schedular['use_lr_schedular'] else None
loss_fn = torch.nn.CrossEntropyLoss()

# Train model on MNIST
loss_hist_train, acc_hist_train = train_loop(model, epochs, optimizer, loss_fn, verbose, train_dataloader, preprocessing_transform, backward_method, l1_regularization_lambda, l2_regularization_lambda, lr_sched)

# Test model on MNIST
loss_hist_test, acc_hist_test = test_loop(model, loss_fn, verbose, test_dataloader, preprocessing_transform)
