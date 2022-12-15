import torch
import numpy as np

# Training Loop ==================== 
def train_loop(model, epochs , optimizer, loss_fn, verbose, train_dataloader, preprocessing_transform, backward_method):
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
            preds = model(X)
            loss = loss_fn(preds, Y)

            # update model
            if backward_method == 'DFA':
                model.loss_distributer(loss,preds)
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

    return loss_hist, acc_hist 




# Testing Loop ==================== 
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