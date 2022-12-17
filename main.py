# https://drive.google.com/drive/u/1/folders/1-ZT3n_axengnhZjYK5AsY8s3oDK1WFnL
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms 

from utils.custom_dataloaders import load_MNIST
from utils.MNIST_model import DNN
from DFA.MNIST_model_DFA import DFA
from utils.train_test_utils import train_loop, test_loop 
import time

# HYPERPARAMETERS -------------------- 
batch_size = 128
epochs = 5
lr = 5e-2
verbose = True
backward_method = "DFA" # possible options "DFA" if not Backprop is used
# ------------------------------------


# Load MNIST dataset
train_dataloader, test_dataloader, preprocessing_transform = load_MNIST(batch_size)
if backward_method == "DFA":
    model = DFA()
else:
    model = DNN()
optimizer = torch.optim.SGD(model.parameters(), lr)
loss_fn = torch.nn.CrossEntropyLoss()

# Train model on MNIST


start = time.time()


loss_hist_train, acc_hist_train = train_loop(model, epochs, optimizer, loss_fn, verbose, train_dataloader, preprocessing_transform, backward_method)

# Test model on MNIST
loss_hist_test, acc_hist_test = test_loop(model, loss_fn, verbose, test_dataloader, preprocessing_transform)


# Add DFA to model architecture ========================================  
# TODO: (B*e)*W = approximated error ~ for Linear layers


# TODO: means to implemet weight alignment metrics
# 