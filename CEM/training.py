import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange, tqdm

def train_network(model, optimizer, train_loader,
                  num_epochs=1, pbar_update_interval=200, print_logs=False):
    '''
    Updates the model parameters (in place) using the given optimizer object.
    Returns `None`.
    '''

    criterion = nn.BCELoss()
    pbar = trange(num_epochs) if print_logs else range(num_epochs)
    total_loss = 0

    for i in pbar:
        for k, batch_data in enumerate(train_loader):
            batch_x = batch_data[:, :-1]
            batch_y = batch_data[:, -1]
            model.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y.unsqueeze(1))
            total_loss += loss.item()
            loss.backward() 
            optimizer.step() 

            if print_logs and k % pbar_update_interval == 0:
                acc = (y_pred.round() == batch_y).sum().float()/(len(batch_y))
                pbar.set_postfix(loss=loss.item(), acc=acc.item())
    print("Total loss: ", total_loss)


