import time
import copy
import torch

def one_epoch(model, data_loader, loss_fn, optimizer=None, verbose=False):
    if optimizer is not None:
        model.train()
    else:
       model.eval()
    device = model.device

    t0 = time.time()
    running_loss = 0
    used_batches = 0

    for i, batch in enumerate(data_loader):
        batch = batch.to(device)

        # Skip single-graph batches only during training (because of BatchNorm)
        if optimizer is not None and batch.num_graphs == 1:
            continue

        if optimizer is not None:
            optimizer.zero_grad()
 
        g0, g1, total = model(batch)
        labels = batch.y.float() # (B,1)
        loss = loss_fn(g0, g1, total, labels)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            running_loss += loss.item()
            used_batches += 1

        if verbose > 0:
            current = i + 1
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{current:>5d}/{used_batches:>5d}]")

    running_loss /= max(used_batches,1)

    if verbose > 0:
        print('epoch time', time.time() - t0)

    return running_loss

def train(model, epochs, train_loader, loss_fn, optimizer, val_loader=None, verbose=True):
    """Train visual dynamics system over multiple epochs."""
    train_losses = []
    # Store copy of best system fit seen so far
    best_model = copy.deepcopy(model)
    best_loss = torch.inf

    if val_loader is not None:
        val_losses = []
    else:
        val_losses = None

    for t in range(epochs):
        if verbose:
            print(f"Epoch {t+1}\n-------------------------------")

        # Update the weights
        train_loss = one_epoch(model, train_loader, loss_fn, optimizer=optimizer, verbose=False)
        # Evaluate the training loss
        train_losses.append(train_loss)
        # Evaluate the val loss
        if val_loader is not None:
            val_loss = one_epoch(model, val_loader, loss_fn, optimizer=None, verbose=False)
            val_losses.append(val_loss)
            if val_loss < best_loss:
                best_model = copy.deepcopy(model)
                best_loss = val_loss
                if verbose:
                    print("Updated best model")
        else:
            if train_loss < best_loss:
                best_model = copy.deepcopy(model)
                best_loss = train_loss
                if verbose:
                    print("Updated best model")
        if verbose:
            print(f"Train Error: Avg loss: {train_loss:>8f}")
            if val_loader is not None:
                print(f"Val Error: Avg loss: {val_loss:>8f}")

    # Return the best model fit and the train/val losses over epochs
    return train_losses, val_losses, best_model

def from_splits(fracs, total):
    n_train = int(fracs[0] * total)
    n_val = int(fracs[1] * total)
    perm = torch.randperm(total)
    split_idx = {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test":  perm[n_train + n_val:]
    }
    return split_idx

def build_custom_loss(reg: float = 0):
    """Write torch custom regularized MSE-like loss which given g0, g1, total
    computes loss as (total - label)^2 + reg * g1^2, like regularized MSE."""
    def loss_fn(g0, g1, total, labels):
        total_flat = total.view(-1)
        labels_flat = labels.view(-1)
        # MSE loss for total = g0 + g1
        mse = torch.mean((total_flat - labels_flat) ** 2)
        # L2 penalty on g1
        reg_term = torch.mean(g1 ** 2) if reg != 0 else 0.0
        loss = mse + reg * reg_term
        return loss

    return loss_fn
