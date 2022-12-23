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


# Training Loop ==================== 
@timer_wrapper("train_loop function")
def train_loop(model, epochs , optimizer, loss_fn, verbose, train_dataloader, preprocessing_transform, backward_method, l1_regularization_lambda, l2_regularization_lambda, lr_sched):
    print(f"==================== Training ====================")
    model.train()
    loss_hist = []
    acc_hist = []
    for epoch in range(1, epochs + 1):
        iter_loss = []
        iter_accuracy_hist = []
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

        cur_epoch_accuracy = np.mean(iter_accuracy_hist)
        # average out metrics
        cur_avg_loss = np.mean(iter_loss)
        loss_hist.append(cur_avg_loss) 
        acc_hist.append(cur_epoch_accuracy)
        if verbose:
            print(f"Epoch: {epoch}, loss: {cur_avg_loss}, accuracy: {cur_epoch_accuracy}")

        # Apply learning rate scheduling
        if lr_sched is not None:
            lr_sched.step()

    return loss_hist, acc_hist 




# Testing Loop ==================== 
@timer_wrapper("test_loop function")
def test_loop(model, loss_fn, verbose, test_dataloader, preprocessing_transform):
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

    return loss_hist, acc_hist 
    
