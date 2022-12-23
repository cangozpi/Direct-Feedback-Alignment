# https://drive.google.com/drive/u/1/folders/1-ZT3n_axengnhZjYK5AsY8s3oDK1WFnL
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms 

from utils.config_parser import read_yaml_config
from utils.custom_dataloaders import load_MNIST
from utils.MNIST_model import DNN
from DFA.MNIST_model_DFA import DFA
from utils.train_test_utils import train_loop, test_loop, set_seed
from torch.utils.tensorboard import SummaryWriter
import datetime
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
    lr_schedular = config['lr_schedular']
    # ------------------------------------ ===========================================

    # Initialize TensorBoard
    log_dir, run_name = "logs/", "cartpole_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)
    # tb_summaryWriter.add_hparams({'lr': 0.1*1, 'bsize': 1}, {}) #TODO: fix this

    # Load MNIST dataset
    train_dataloader, test_dataloader, preprocessing_transform = load_MNIST(batch_size)
    if backward_method == "DFA":
        model = DFA(p_drop, use_BatchNorm1D, use_LayerNorm1D)
    else:
        model = DNN()
    optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay = weight_decay)
    lr_sched = torch.optim.lr_scheduler.ConstantLR(optimizer, factor = float(lr_schedular['factor']), total_iters = lr_schedular['total_iters']) if lr_schedular['use_lr_schedular'] else None
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train model on MNIST
    loss_hist_train, acc_hist_train = train_loop(model, epochs, optimizer, loss_fn, verbose, train_dataloader, preprocessing_transform, backward_method, l1_regularization_lambda, l2_regularization_lambda, lr_sched, tb_summaryWriter)

    # Test model on MNIST
    loss_hist_test, acc_hist_test = test_loop(model, loss_fn, verbose, test_dataloader, preprocessing_transform)
