import random
import torch
import numpy as np
from DFA.layers import DFA_Linear


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)

# Wrapper function that times how long the function execution took
def timer_wrapper(func_name):
    def func_wrapper(func):
        import time
        def wrapper(*args, **kwargs):
            start_time = time.time()
            return_val = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func_name} took {(end_time - start_time):.3f} seconds to execute.")
            return return_val

        return wrapper
    return func_wrapper


# Implements Early Stopping  #TODO: MNIST training does not have validation dataset so currently it is not added to train_loop 
#TODO: MNIST training does not have validation dataset so currently it is not added to train_loop
class EarlyStopping:
    def __init__(self, tolerance = 5, min_delta = 0):
        """
        Used as follows:
            In your training loop in between epochs put the following:
                # early stopping
                early_stopping(epoch_train_loss, epoch_validate_loss)
                if early_stopping.early_stop:
                    print(f"Early Stopping stopped at epoch: {epoch}")
                    break
        """
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True


# Training Loop ==================== 
@timer_wrapper("train_loop function")
def train_loop(model, epochs , optimizer, loss_fn, verbose, train_dataloader, preprocessing_transform, backward_method, l1_regularization_lambda, l2_regularization_lambda, lr_sched, tb_summaryWriter):
    print(f"==================== Training ====================")
    model.train()
    loss_hist = []
    acc_hist = []
    for epoch in range(1, epochs + 1):
        iter_loss = []
        iter_accuracy_hist = []
        iter_model_param_dict = {} # Records model parameters (weights & grads) through iterations
        for X, Y in train_dataloader:
            model.zero_grad()
            # preprocess image inputs
            X = preprocessing_transform(X)
            preds = model(X) # [B, output_dim]
            loss = loss_fn(preds, Y)


            # Calculate L1 regularization loss
            l1_regularization_loss = 0
            l1_regularization_layer_weights = []
            for m in model.modules():
                if isinstance(m, DFA_Linear): # Only DFA_Linear layer's weights are taken into account for L1 regularization
                    cur_layer_weights = m.weight # obtain current layer's weight tensor
                    cur_layer_weights = cur_layer_weights.reshape(-1) # Flatten weights
                    l1_regularization_layer_weights.append(cur_layer_weights)


            l1_regularization_layer_weights = torch.cat(l1_regularization_layer_weights) # --> [num_linear_layers, Flatten(layer_weight)]
            l1_regularization_loss += ( l1_regularization_lambda * torch.norm(l1_regularization_layer_weights, 1))

            # Calculate L2 regularization loss
            l2_regularization_loss = 0
            l2_regularization_layer_weights = []
            for m in model.modules():
                if isinstance(m, DFA_Linear): # Only DFA_Linear layer's weights are taken into account for L1 regularization
                    cur_layer_weights = m.weight # obtain current layer's weight tensor
                    cur_layer_weights = cur_layer_weights.reshape(-1) # Flatten weights
                    l2_regularization_layer_weights.append(cur_layer_weights)


            l2_regularization_layer_weights = torch.cat(l2_regularization_layer_weights) # --> [num_linear_layers, Flatten(layer_weight)]
            l2_regularization_loss += (l2_regularization_lambda * torch.square(torch.norm(l2_regularization_layer_weights, 2)))

            # Add L1 and L2 regularizations to the loss
            loss += l1_regularization_loss + l2_regularization_loss


            # update model
            if backward_method == 'DFA':
                model.loss_distributer(loss, preds)
                loss.backward()
            else:
                loss.backward()

            optimizer.step()

            # record metrics
            iter_loss.append(loss.cpu().detach())

            pred_indices = torch.argmax(preds.cpu().detach(), dim=-1)
            iter_accuracy = torch.sum(pred_indices == Y) / X.shape[0]
            iter_accuracy_hist.append(iter_accuracy)

            for name, param in model.named_parameters(): # Record model parameters (weights & grad)
                if hasattr(param, 'grad') and (param.grad is not None):
                    if name not in iter_model_param_dict:
                        iter_model_param_dict[name] = [param]
                        iter_model_param_dict[name+".grad"] = [param.grad]
                    else:
                        iter_model_param_dict[name].append(param)
                        iter_model_param_dict[name+".grad"].append(param.grad)

        cur_epoch_accuracy = np.mean(iter_accuracy_hist)
        # average out metrics
        cur_avg_loss = np.mean(iter_loss)
        loss_hist.append(cur_avg_loss) 
        acc_hist.append(cur_epoch_accuracy)
        if verbose and (lr_sched is not None):
            print(f"Epoch: {epoch}, loss: {cur_avg_loss}, accuracy: {cur_epoch_accuracy}, lr: {lr_sched.get_last_lr()[-1]}")
        elif verbose:
            print(f"Epoch: {epoch}, loss: {cur_avg_loss}, accuracy: {cur_epoch_accuracy}")
        
        # Log to Tensorboard
        tb_summaryWriter.add_scalar("Training Loss", loss_hist[-1], epoch)
        tb_summaryWriter.add_scalar("Training Accuracy", acc_hist[-1], epoch)
        if lr_sched is not None:
            tb_summaryWriter.add_scalar("Training Learning Rate", lr_sched.get_last_lr()[-1], epoch)

        # Log iterations weights and gradients 
        for name, param in iter_model_param_dict.items():
            # average out recorded values over the iterations 
            param = torch.stack(param) # --> [num_iter, weight_dim1, weight_dim2]
            avg_param = torch.mean(param, dim=0)
            tb_summaryWriter.add_histogram(name, avg_param, epoch)

        # Apply learning rate scheduling
        if lr_sched is not None:
            lr_sched.step()


    return loss_hist, acc_hist 




# Testing Loop ==================== 
@timer_wrapper("test_loop function")
def test_loop(model, loss_fn, verbose, test_dataloader, preprocessing_transform, tb_summaryWriter):
    print(f"==================== Testing ====================")
    model.eval()
    loss_hist = []
    acc_hist = []

    iter_loss = []
    iter_accuracy_hist = []
    for X, Y in test_dataloader:
        # preprocess image inputs
        X = preprocessing_transform(X)
        with torch.no_grad():
            preds = model(X)
        loss = loss_fn(preds, Y)

        # record metrics
        iter_loss.append(loss.cpu().detach())

        pred_indices = torch.argmax(preds.cpu().detach(), dim=-1)
        iter_accuracy = torch.sum(pred_indices == Y) / X.shape[0]
        iter_accuracy_hist.append(iter_accuracy)

    cur_epoch_accuracy = np.mean(iter_accuracy_hist)
    # average out metrics
    cur_avg_loss = np.mean(iter_loss)
    loss_hist.append(cur_avg_loss) 
    acc_hist.append(cur_epoch_accuracy)
    if verbose:
        print(f"Test_loss: {cur_avg_loss}, Test_accuracy: {cur_epoch_accuracy}")
    # Log to Tensorboard
    tb_summaryWriter.add_scalar("Test Loss", cur_avg_loss, 1)
    tb_summaryWriter.add_scalar("Test Accuracy", cur_epoch_accuracy, 1)

    return loss_hist, acc_hist 
    
