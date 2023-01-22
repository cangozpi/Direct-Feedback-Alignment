# https://drive.google.com/drive/u/1/folders/1-ZT3n_axengnhZjYK5AsY8s3oDK1WFnL
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms 

from utils.config_parser import read_yaml_config, get_tensorboard_hparam_dict
from utils.custom_dataloaders import load_MNIST
from utils.MNIST_model import DNN
from DFA.MNIST_model_DFA import DFA
from utils.train_test_utils import train_loop, test_loop, set_seed
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import copy
import time

if __name__ == "__main__":
    set_seed(42) # Set seed for reproducibility reasons

    # Read in parameters from config.yaml ===========================================
    config_path = 'config.yaml'
    config = read_yaml_config(config_path)

    # HYPERPARAMETERS -------------------- 
    batch_size = int(config['batch_size'])
    epochs = int(config['epochs'])
    lr = float(config['lr'])
    verbose = config['verbose']
    backward_method = config['backward_method']
    # Regularization Methods ------------------------------------
    p_drop = float(config['p_drop'])
    use_BatchNorm1D = config['use_BatchNorm1D']
    use_LayerNorm1D = config['use_LayerNorm1D']
    l1_regularization_lambda = float(config['l1_regularization_lambda'])
    l2_regularization_lambda = float(config['l2_regularization_lambda'])
    weight_decay = float(config['weight_decay'])
    lr_schedular = copy.deepcopy(config['lr_schedular'])
    # ------------------------------------ ===========================================

    # Initialize TensorBoard
    log_dir, run_name = "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
    tb_summaryWriter = SummaryWriter(os.path.join(log_dir, run_name)) 

    # Load MNIST dataset
    train_dataloader, test_dataloader, preprocessing_transform = load_MNIST(batch_size)
    if backward_method == "DFA":
        model = DFA(p_drop, use_BatchNorm1D, use_LayerNorm1D)
    else:
        model = DNN(p_drop, use_BatchNorm1D, use_LayerNorm1D)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay = weight_decay)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size = int(lr_schedular['step_size']), gamma = float(lr_schedular['gamma'])) if lr_schedular['use_lr_schedular'] else None
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train model on MNIST
    loss_hist_train, acc_hist_train = train_loop(model, epochs, optimizer, loss_fn, verbose, train_dataloader, preprocessing_transform, backward_method, l1_regularization_lambda, l2_regularization_lambda, lr_sched, tb_summaryWriter)

    # Test model on MNIST
    loss_hist_test, acc_hist_test = test_loop(model, loss_fn, verbose, test_dataloader, preprocessing_transform, tb_summaryWriter)


    # Record Hyperparameters on TensorBoard
    hparam_dict = get_tensorboard_hparam_dict(config, lr_schedular)
    tb_summaryWriter.add_hparams(hparam_dict, {"hparam/dummy_metric": -1},
                   run_name=os.path.join(os.path.dirname(os.path.realpath(__file__)), log_dir, run_name))
    tb_summaryWriter.close()
